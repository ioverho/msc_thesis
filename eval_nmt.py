import os
import pickle
from datetime import datetime
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import hydra
from omegaconf import DictConfig, OmegaConf
import pycountry

from morphological_tagging.pipelines import UDPipe2Pipeline
from nmt_adapt.data.corpus import ParallelTreebankCorpus
from nmt_adapt.inverse_index import InverseIndex
from nmt_adapt.sample import generate_samples
from nmt_adapt.metrics import token_metrics, morphological_metrics, entropy
from utils.experiment import HidePrints, Timer, set_seed, set_deterministic
from utils.tokenizers import MosesTokenizerWrapped

EVAL_STATS_LOC = "./nmt_adapt/eval"

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def save_stats_file(stats_dict, fp):

    stats_dict = dict(stats_dict)
    for m_tag in stats_dict.keys():
        stats_dict[m_tag] = dict(stats_dict[m_tag])

    with open(fp, "wb") as f:
        pickle.dump(stats_dict, f)


@hydra.main(config_path="./nmt_adapt/config", config_name="nmt_evaluation")
def eval_nmt(config):

    timer = Timer()

    # *==========================================================================
    # *Config reading
    # *==========================================================================
    if config["print_hparams"]:
        print(50 * "+")
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print(OmegaConf.to_yaml(config))
        print(50 * "+")
    else:
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print("Loaded.")

    # *==========================================================================
    # *Experiment
    # *==========================================================================
    print(f"\n{timer.time()} | SETUP")

    # == Device
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    # *==============================================================================
    # *Data Import
    # *==============================================================================
    print(f"\n{timer.time()} | IMPORTING DATASETS")
    par_data = ParallelTreebankCorpus.load(
        f"./nmt_adapt/data/corpora/annotated_test_sets/{config['src_lang']}_{config['tgt_lang']}_test.pickle"
    )

    # *==========================================================================
    # *Inverted Index
    # *==========================================================================
    print(f"\n{timer.time()} | BUILDING INVERTED INDEX")
    index = InverseIndex(
        par_data,
        filter_level=config["index"]["filter_level"],
        index_level=config["index"]["index_level"],
    )
    index.filter(filter_vals=set(config["index"]["index_filter_vals"]))
    index.reduce(par_data, max_samples=config["index"]["max_tag_samples"])

    print(f"Index has {len(index.keys())} keys, and {len(index)} values.")

    # *==============================================================================
    # *Model Import
    # *==============================================================================
    print(f"\n{timer.time()} | INITIALIZING MODELS")
    # Import the model and prepare for evaluation

    model_name = (
        config["model_name"]
        if "model_name" in set(config.keys())
        else f"Helsinki-NLP/opus-mt-{config['src_lang']}-{config['tgt_lang']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, enable_sampling=False, nbest_size=0,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.require_grad = False
    model = model.to(device)

    vocab_size = model._modules["lm_head"].out_features

    # Import the tagging pipeline
    lang_full = pycountry.languages.get(alpha_2=config["tgt_lang"]).name
    expected_pipeline_path = (
        f"./morphological_tagging/pipelines/UDPipe2_{lang_full}_merge.ckpt"
    )
    print(f"Looking for pipeline in {expected_pipeline_path}")

    tagger = UDPipe2Pipeline.load(
        expected_pipeline_path, tokenizer=MosesTokenizerWrapped(lang=config["tgt_lang"])
    )
    tagger = tagger.to(device)

    # *==============================================================================
    # *Eval Loop
    # *==============================================================================
    print(f"\n{timer.time()} | EVALUATION LOOP")

    src_prefix = config["prefix"] if "prefix" in config.keys() else ""
    tgt_prefix = tokenizer.pad_token

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    stats_fp = (
        f"{EVAL_STATS_LOC}/{timestamp}_{config['src_lang']}_{config['tgt_lang']}.pickle"
    )

    morph_tag_stats = defaultdict(lambda: defaultdict(list))
    for i, (morph_tag_set, (sent_id, t)) in enumerate(iter(index)):
        src_text = src_prefix + par_data[sent_id].parallel_text

        ref_context = par_data[sent_id].tokens[:t]
        ref_label = par_data[sent_id].tokens[t : t + 1]

        src = tokenizer(src_text, padding=True, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            tgt_context = tokenizer(
                [[tgt_prefix] + (ref_context if len(ref_context) > 0 else [""])],
                return_tensors="pt",
                is_split_into_words=True,
            )

            tgt = tokenizer(
                ref_label, padding=False, return_tensors="pt", is_split_into_words=True,
            )

        model_kwargs = (
            {"decoder_input_ids": tgt_context["input_ids"].to(device)}
            if t != 0
            else dict()
        )

        with torch.no_grad():
            samples = model.generate(
                input_ids=src["input_ids"].to(device),
                attention_mask=src["attention_mask"].to(device),
                do_sample=True,
                early_stopping=True,
                num_beams=1,
                num_return_sequences=config["n_samples"],
                max_length=tgt_context["input_ids"].size(-1)
                + tgt["input_ids"].size(-1)
                + config["max_over_T"],
                **config["sampling_method_kwargs"],
                **model_kwargs,
            )

        samples = samples.detach().to("cpu")

        with tokenizer.as_target_tokenizer():
            textual_samples = tokenizer.batch_decode(samples, skip_special_tokens=True,)

        sampled_tokens = []
        for seq in textual_samples:
            tokenized_samples = tagger.tokenizer(seq)

            if len(tokenized_samples) == 0:
                sampled_tokens.append("_")

            t_ = min(len(tokenized_samples) - 1, t)

            sampled_tokens.append(tokenized_samples[t_])

        sample_counts = sorted(
            list(Counter(sampled_tokens).items()), key=lambda x: x[1], reverse=True
        )

        with HidePrints():
            metrics = token_metrics(ref_label[0], sample_counts, tokenizer)
            metrics |= morphological_metrics(
                par_data[sent_id], sample_counts, t, tagger
            )
            metrics |= {"entropy": entropy(sample_counts)}

        morph_tag_stats[morph_tag_set][par_data[sent_id].lemma_scripts[t]].append(
            metrics
        )

        if (i + 1) % config["save_every_n"] == 0 or i == 0:
            print(f"\t{timer.time()} | Record {i}")
            try:
                print("\t", src_text, sep="")
                print(
                    "\t",
                    ref_context[max(0, len(ref_context) - 5) :],
                    ref_label[0],
                    sample_counts[0:2],
                    sep="",
                )

            except UnicodeEncodeError:
                print("Can't print, but trust me, it's still working.")

            save_stats_file(morph_tag_stats, stats_fp)

    # ==============================================================================
    # Saving
    # ==============================================================================
    save_stats_file(morph_tag_stats, stats_fp)

    print(f"Finished.")
    timer.end()


if __name__ == "__main__":

    eval_nmt()
