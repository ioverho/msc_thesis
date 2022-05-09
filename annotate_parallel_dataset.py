import os
import math

# 3rd Party
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pycountry

# User-defined
from morphological_tagging.pipelines import UDPipe2Pipeline
from nmt_adapt.data.corpus import ParallelTreebankCorpus, AnnotatedSentence
from utils.tokenizers import MosesTokenizerWrapped
from utils.experiment import set_seed, set_deterministic, Timer, progressbar

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./nmt_adapt/config", config_name="data_annotation")
def annotate(config: DictConfig):
    """Annotation loop.

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

    print(f"\n{timer.time()} | DATA SETUP")
    par_data = ParallelTreebankCorpus(
        src_lang=config["src_lang"], tgt_lang=config["tgt_lang"]
    )

    if config["parallel_dataset"].get("hf", False):
        print("Loading in a Huggingface dataset.")
        par_data.load_hf_dataset(**config["parallel_dataset"]["hf"])

    if config["parallel_dataset"].get("tatoeba", False):
        print("Loading in the HelsinkiNLP/Tatoeba testsets.")
        par_data.load_tatoeba_dataset()

    if config["parallel_dataset"].get("flores", False):
        print("Loading in the Flores101 testset.")
        par_data.load_flores_dataset(**config["parallel_dataset"]["flores"])

    print(f"\n{timer.time()} | MODEL IMPORT")
    lang_full = pycountry.languages.get(alpha_2=config["tgt_lang"]).name
    expected_pipeline_path = (
        f"./morphological_tagging/pipelines/UDPipe2_{lang_full}_merge.ckpt"
    )
    print(f"Looking for pipeline in {expected_pipeline_path}")

    pipeline = UDPipe2Pipeline.load(expected_pipeline_path)

    pipeline.tagger.eval()
    for param in pipeline.parameters():
        param.requires_grad = False
    pipeline = pipeline.to(device)

    pipeline.add_tokenizer(MosesTokenizerWrapped(lang=config["tgt_lang"]))

    print(f"\n{timer.time()} | ANNOTATING")
    batch_size = config["batch_size"]

    n_batches = math.ceil(len(par_data.parallel_dataset) / batch_size)
    for i in progressbar(range(n_batches), prefix="Annotating", size=80):

        actual_batch_size = (
            min((i + 1) * batch_size, len(par_data.parallel_dataset)) - i * batch_size
        )

        batch = [
            [
                par_data.parallel_dataset[ii]["source"],
                par_data.parallel_dataset[ii]["id"],
                par_data.parallel_dataset[ii][config["src_lang"]],
                pipeline.tokenizer(par_data.parallel_dataset[ii][config["tgt_lang"]]),
            ]
            for ii in range(i * batch_size, i * batch_size + actual_batch_size,)
        ]

        sources, ids, src_text, tgt_tokens = list(map(list, zip(*batch)))
        lemmas, lemma_scripts, morph_tags, morph_cats = pipeline(
            tgt_tokens, is_pre_tokenized=True
        )

        annotated_sents = [
            AnnotatedSentence(
                source_file=sources[ii],
                id=ids[ii],
                parallel_text=src_text[ii],
                tokens=tgt_tokens[ii],
                lemmas=lemmas[ii],
                lemma_scripts=lemma_scripts[ii],
                morph_tags=morph_tags[ii],
                morph_cats=morph_cats[ii],
            )
            for ii in range(actual_batch_size)
        ]

        par_data.extend(annotated_sents)

    print(f"Annotated {len(par_data)} sentences.")

    print(f"\n{timer.time()} | SAVING ANNOTATED PARALLEL DATASET FILE")
    save_path = (
        f"./nmt_adapt/data/corpora/annotated_test_sets/{par_data.src_lang}"
        + f"_{par_data.tgt_lang}_{config['affix']}.pickle"
    )
    print(f"Saving to {save_path}")
    par_data.save(fp=save_path)

    timer.end()

    return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS

    annotate()
