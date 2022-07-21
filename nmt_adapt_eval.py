import os
import pickle
from datetime import datetime
from collections import defaultdict, Counter

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig, OmegaConf
import pycountry

from morphological_tagging.pipelines import UDPipe2Pipeline
from nmt_adapt.data.corpus_functional import load_custom_dataset
from nmt_adapt.inverse_index import InverseIndexv2
from nmt_adapt.metrics import token_metrics, morphological_metrics, entropy
from utils.experiment import HidePrints, Timer, set_seed, set_deterministic
from utils.tokenizers import MosesTokenizerWrapped

INDICES_LOC = "./nmt_adapt/data/indices/"
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
    """It takes a model, a dataset, and an index, and then it samples from the model, and then it computes
    a bunch of metrics on the samples

    Parameters
    ----------
    config
        The configuration file that hydra uses to run the experiment.

    """

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

    src_lang = pycountry.languages.get(name=config["src_lang"])
    tgt_lang = pycountry.languages.get(name=config["tgt_lang"])

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
    par_data = load_custom_dataset(
        src_lang.name,
        tgt_lang.name,
        dataset_name=config["data"]["dataset_name"],
        split=config["data"].get("split", None),
        source=None,
    )
    print(f"Dataset length: {len(par_data)}")

    # *==========================================================================
    # *Inverted Index
    # *==========================================================================
    print(f"\n{timer.time()} | BUILDING INVERTED INDEX")
    if config["index"].get("fp", None) is None:
        print("\nBuilding index.")
        index = InverseIndexv2(
            par_data=par_data,
            index_level=config["index"]["index_level"],
            filter_level=config["index"]["filter_level"],
        )
        print("Built index.")
        print(index.length_str)
        index.reduce(**config["index"]["reduce"])
        print(index.length_str)
        print(f"Document coverage of {index.coverage}/{len(par_data)}")

        index.save(
            INDICES_LOC
            + config["data"]["dataset_name"]
            + f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}.pickle"
        )

    else:
        index = InverseIndexv2.load(config["index"]["fp"])
        print(f"Loaded index from {config['index']['fp']}.")
        print(f"Index has {len(index.keys())} keys, and {len(index)} values.")

    # *==============================================================================
    # *Model Import
    # *==============================================================================
    print(f"\n{timer.time()} | INITIALIZING MODELS")
    # Import the model and prepare for evaluation

    model_name = (
        config["model_name"]
        if "model_name" in set(config.keys())
        else f"Helsinki-NLP/opus-mt-{src_lang.alpha_2}-{tgt_lang.alpha_2}"
    )

    model_config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, enable_sampling=False, nbest_size=0,
    )

    if config.get("checkpoint_path", None) is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=model_config,
            )

    else:
        model  = AutoModelForSeq2SeqLM.from_pretrained(
            config["checkpoint_path"],
            config=model_config,
            )

    model.eval()
    for param in model.parameters():
        param.require_grad = False
    model = model.to(device)

    # Import the tagging pipeline
    expected_pipeline_path = (
        f"./morphological_tagging/pipelines/UDPipe2_{tgt_lang.name}_merge.ckpt"
    )
    print(f"Looking for pipeline in {expected_pipeline_path}")

    tagger = UDPipe2Pipeline.load(
        expected_pipeline_path, tokenizer=MosesTokenizerWrapped(lang=tgt_lang.alpha_2)
    )
    tagger = tagger.to(device)

    # *==============================================================================
    # *Eval Loop
    # *==============================================================================
    print(f"\n{timer.time()} | EVALUATION LOOP")

    src_prefix = config["prefix"] if "prefix" in config.keys() else ""
    tgt_prefix = tokenizer.pad_token

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    stats_fp = f"{EVAL_STATS_LOC}/{timestamp}_{config['data']['dataset_name']}"
    stats_fp += f"_{config['data']['split']}" if config['data'].get('split', None) is not None else ""
    stats_fp += f"_{config['note']}" if config.get('note', None) is not None else ""
    stats_fp += f"_{src_lang.name}_{tgt_lang.name}.pickle"

    morph_tag_stats = defaultdict(lambda: defaultdict(list))
    for i, (morph_tag_set, (sent_id, t)) in enumerate(iter(index)):
        src_text = src_prefix + par_data[sent_id]["src_text"]

        # Make **certain** the index still points to the correct examples
        assert set(par_data[sent_id]["morph_tags"][t]) == set(
            morph_tag_set.morph_tag_set
        ), f"({sent_id}, {t}) is {set(par_data[sent_id]['morph_tags'][t])} not {set(morph_tag_set)}"

        ref_context = par_data[sent_id]["tgt_tokens"][:t]
        ref_label = par_data[sent_id]["tgt_tokens"][t : t + 1]

        src = tokenizer(src_text, padding=True, truncation=True, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            tgt_context = tokenizer(
                [[tgt_prefix] + (ref_context if len(ref_context) > 0 else [""])],
                return_tensors="pt",
                is_split_into_words=True,
            )

            tgt = tokenizer(
                ref_label, padding=False, return_tensors="pt", is_split_into_words=True,
            )

        if tgt_context["input_ids"].size(1) >= model_config.max_length:
            print(">>Skipping<< Target-side context is too long for model.")
            continue

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

            try:
                sampled_tokens.append(tokenized_samples[t_])

            except IndexError:
                continue

        if len(sampled_tokens) == 0:
            print(">>Skipping<< Malformed generations.")
            continue

        sample_counts = sorted(
            list(Counter(sampled_tokens).items()), key=lambda x: x[1], reverse=True
        )

        with HidePrints():
            metrics = token_metrics(ref_label[0], sample_counts, tokenizer)
            metrics |= morphological_metrics(
                par_data[sent_id], sample_counts, t, tagger
            )
            metrics |= {"entropy": entropy(sample_counts)}

        morph_tag_stats[morph_tag_set][par_data[sent_id]["lemmas"][t]].append(metrics)

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
