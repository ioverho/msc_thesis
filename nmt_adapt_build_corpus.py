import os
import math

# 3rd Party
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# User-defined
from morphological_tagging.pipelines import UDPipe2Pipeline
from nmt_adapt.data.corpus_functional import load_custom_dataset
from utils.tokenizers import MosesTokenizerWrapped
from utils.experiment import set_seed, set_deterministic, Timer

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CORPORA_LOC = "./nmt_adapt/data/corpora/"


@hydra.main(config_path="./nmt_adapt/config", config_name="annotate_parallel_data")
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
    par_data = load_custom_dataset(**config["data"])

    print(f"\n{timer.time()} | MODEL IMPORT")
    expected_pipeline_path = (
        f"./morphological_tagging/pipelines/UDPipe2_{config['data']['tgt_lang']}_merge.ckpt"
    )
    print(f"Looking for pipeline in {expected_pipeline_path}")

    pipeline = UDPipe2Pipeline.load(expected_pipeline_path)

    pipeline.tagger.eval()
    for param in pipeline.parameters():
        param.requires_grad = False
    pipeline = pipeline.to(device)

    pipeline.add_tokenizer(MosesTokenizerWrapped(lang=config["data"]["tgt_lang"]))

    def annotate(batch):

        lemmas, lemma_scripts, morph_tags, morph_cats = pipeline.forward(
            batch["tgt_tokens"], is_pre_tokenized=True
        )

        return {
            "lemmas": lemmas,
            "lemma_scripts": lemma_scripts,
            "morph_tags": morph_tags,
            "morph_cats": morph_cats,
        }

    print(f"\n{timer.time()} | ANNOTATING")

    par_data = par_data.map(annotate, batched=True, batch_size=config["batch_size"])

    print(f"\n{timer.time()} | SAVING ANNOTATED PARALLEL DATASET FILE")
    par_data.save_to_disk(f"{CORPORA_LOC}/{par_data.info.config_name}")

    timer.end()

    return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS

    annotate()
