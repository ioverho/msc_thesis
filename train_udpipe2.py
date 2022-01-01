# *==============================================================================
# *Package import
# *==============================================================================
import os
import yaml
import argparse
import warnings
from shutil import copyfile

# 3rd Party
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# User-defined
from morphological_tagging.data.corpus import get_conllu_files, DocumentCorpus, FastText
from morphological_tagging.models.udpipe2 import UDPipe2

from utils.experiment import find_version, set_seed, set_deterministic, Timer
from utils.errors import ConfigurationError

CHECKPOINT_DIR = "./morphological_tagging/checkpoints"


def train(args):
    """Train loop.

    """

    timer = Timer()

    # *==========================================================================
    # *Config reading
    # *==========================================================================
    with open(args.config_file_path, "r") as f:
        config = yaml.safe_load(f)

    print(50 * "+")
    print("HYPER-PARAMETERS")
    print(yaml.dump(config))
    print(50 * "+")

    # *==========================================================================
    # *Experiment
    # *==========================================================================
    print(f"\n{timer.time()} | EXPERIMENT SETUP")

    full_name = f"{config['run']['experiment_name']}_{config['data']['language']}_{config['data']['name']}"

    # == Version
    # ==== ./checkpoints/data_version/version_number
    full_version, experiment_dir, version = find_version(
        full_name,
        CHECKPOINT_DIR,
        debug=config["run"]["debug"]
    )

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    copyfile(
        args.config_file_path,
        f"{CHECKPOINT_DIR}/{full_version}/{os.path.split(args.config_file_path)[-1]}",
    )

    print(f"Saving to {CHECKPOINT_DIR}/{full_version}")

    # == Device
    use_cuda = config["run"]["gpu"] or config["run"]["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # == Reproducibility
    set_seed(config["run"]["seed"])
    if config["run"]["deterministic"]:
        set_deterministic()

    # *==========================================================================
    # * Logging & Callbacks
    # *==========================================================================
    # == Logging
    print(f"\n{timer.time()} | LOGGER SETUP")
    if config["logging"]["logger"].lower() == "tensorboard":
        # os.path.join(save_dir, name, version)
        logger = TensorBoardLogger(
            save_dir=f"{CHECKPOINT_DIR}",
            name=f"{experiment_dir}",
            version=f"version_{version}",
            #**config["logging"]["logger_kwargs"],
        )

    elif config["logging"]["logger"].lower() in ["wandb", "weightsandbiases"]:
        logger = WandbLogger(
            project="morphological_tagging",
            save_dir=f"{CHECKPOINT_DIR}/{full_version}",
            name=f"{experiment_dir}_v{version}",
            version=version,
            **config["logging"]["logger_kwargs"],
        )
        logger.experiment.config.update(config, allow_val_change=True)

    else:
        raise ConfigurationError("Logger not recognized.")

    # == Callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{CHECKPOINT_DIR}/{full_version}/checkpoints",
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["monitor_mode"],
        auto_insert_metric_name=True,
        save_last=True,
    )
    callbacks += [checkpoint_callback]

    if config["model"].get("scheduler_name", False):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks += [lr_monitor]

    device_monitor = DeviceStatsMonitor()
    callbacks += [device_monitor]

    prog_bar = TQDMProgressBar(refresh_rate=config['run']['prog_bar_refresh_rate'])
    callbacks += [prog_bar]

    # *==========================================================================
    # * Dataset
    # *==========================================================================
    print(f"\n{timer.time()} | DATA SETUP")
    files = get_conllu_files(
        language=config["data"]["language"],
        name=config["data"]["name"],
        splits=config["data"]["splits"],
    )

    corpus = DocumentCorpus(
        batch_first=config["data"]["batch_first"],
        sorted=config["data"]["sorted"]
    )

    for (fp, t_name, split) in files:
        corpus.parse_tree_file(fp, t_name, split)
    corpus.setup()

    corpus.set_lemma_tags()

    if config["data"]["FastText"]:
        corpus.add_word_embs(
            FastText,
            language=config["data"]["language"],
            cache="./morphological_tagging/data/pretrained_vectors",
        )

    print(corpus)

    train_corpus = Subset(corpus, corpus.splits["train"])
    train_loader = DataLoader(
        train_corpus,
        batch_size=config["misc_hparams"]["batch_size"],
        shuffle=False,
        collate_fn=corpus.collate_batch,
    )

    valid_corpus = Subset(corpus, corpus.splits["dev"])
    valid_loader = DataLoader(
        valid_corpus,
        batch_size=config["misc_hparams"]["batch_size"],
        shuffle=False,
        collate_fn=corpus.collate_batch,
    )

    test_corpus = Subset(corpus, corpus.splits["test"])
    test_loader = DataLoader(
        test_corpus,
        batch_size=config["misc_hparams"]["batch_size"],
        shuffle=False,
        collate_fn=corpus.collate_batch,
    )

    # *==========================================================================
    # * Model
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")
    model = UDPipe2(
        char_vocab=corpus.char_vocab,
        token_vocab=corpus.token_vocab,
        unk_token=corpus.unk_token,
        pad_token=corpus.pad_token,
        pretrained_embedding_dim=corpus.pretrained_embeddings_dim,
        n_lemma_scripts=len(corpus.script_counter),
        n_morph_tags=len(corpus.morph_tag_vocab),
        n_morph_cats=len(corpus.morph_cat_vocab),
        **config["model"],
    )

    # *==========================================================================
    # * Train
    # *==========================================================================
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=1 if use_cuda else 0,
        deterministic=config["run"]["deterministic"],
        fast_dev_run=(int(config["run"]["fdev_run"])
                      if config["run"]["fdev_run"] > 0 or config["run"]["fdev_run"]
                      else False),
        weights_summary="top",
        **config["trainer"],
    )

    trainer.logger._default_hp_metric = None

    print(f"\n{timer.time()} | TRAINING")

    trainer.fit(model, train_dataloaders=[train_loader], val_dataloaders=[valid_loader])

    # *##########
    # * TESTING #
    # *##########
    if not (config["run"]["fdev_run"] > 0 or config["run"]["fdev_run"]):
        # If in fastdev mode, won't save a model
        # Would otherwise throw a 'PermissionError: [Errno 13] Permission denied: ...'

        print("\nTESTING")
        print(f"LOADING FROM {trainer.checkpoint_callback.best_model_path}")
        model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        model.freeze()
        model.eval()

        test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

        timer.end()

        return test_result

    else:
        timer.end()

        return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument("--config_file_path", default="./config/udpipe2.yaml", type=str)

    args = parser.parse_args()

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings("ignore", message=r".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", message=r".*does not have many workers which may be a bottleneck.*"
    )
    warnings.filterwarnings("ignore", message=r".*GPU available but not used .*")
    warnings.filterwarnings("ignore", message=r".*shuffle=True")
    warnings.filterwarnings("ignore", message=r".*Trying to infer .*")

    train(args)
