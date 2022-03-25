import os
import warnings
import yaml
from pathlib import Path

# 3rd Party
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import pycountry

# User-defined
from nmt_evaluation.data.corpus import ParallelTreebankCorpus
from morphological_tagging.models import UDPipe2
from utils.experiment import set_seed, set_deterministic, Timer, progressbar, HidePrints
from utils.errors import ConfigurationError

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./nmt_evaluation/config", config_name="data_annotation")
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

    # *==========================================================================
    # * Dataset
    # *==========================================================================
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

    par_data.load_vocabs_from_treebankdatamodule_checkpoint(
        fp=config["treebank_data_module_file_path"]
    )

    # *==========================================================================
    # * Model
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")

    if config["model"]["architecture"].lower() == "udpipe2":
        if config["model"].get("tagger_checkpoint_file_path", False):
            fp = config["model"]["tagger_checkpoint_file_path"]
        else:
            lang_full = pycountry.languages.get(alpha_2=config["tgt_lang"]).name
            fp = f"./morphological_tagging/checkpoints/UDPipe2_{lang_full}_merge/version_1/checkpoints/last.ckpt"

        model = UDPipe2.load_from_checkpoint(fp, map_location=device)

    else:
        raise NotImplementedError(
            "Architecture {config['architecture'].lower()} not recognized."
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    # *==========================================================================
    # * Annotation
    # *==========================================================================
    print(f"\n{timer.time()} | ANNOTATING")
    data_loader = DataLoader(
        par_data.parallel_dataset,
        batch_size=config["batch_size"],
        drop_last=False,
        shuffle=False,
        collate_fn=par_data.collate_for_tagger,
    )

    for ids, sources, par_texts, batch in progressbar(
        data_loader, prefix="Annotating", size=80
    ):

        lemma_preds, morph_preds = model.pred_step(batch)

        with HidePrints():
            par_data.parse_sent_from_preds(
                ids, sources, par_texts, batch, lemma_preds, morph_preds
            )

    print(f"Annotated {len(par_data)} sentences.")

    print(f"\n{timer.time()} | SAVING ANNOTATED PARALLEL DATASET FILE")
    save_path = (
        f"./nmt_evaluation/data/corpora/{par_data.src_lang}"
        + f"_{par_data.tgt_lang}.pickle"
    )
    print(f"Saving to {save_path}")
    par_data.save(fp=save_path)

    timer.end()

    return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS

    annotate()
