# *==============================================================================
# *Package import
# *==============================================================================
import os
import argparse
import warnings
import yaml

# 3rd Party
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# User-defined
from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.models.udify import UDIFY
from utils.experiment import find_version, set_seed, set_deterministic, Timer
from utils.errors import ConfigurationError

CHECKPOINT_DIR = "./morphological_tagging/checkpoints"

dotenv.load_dotenv(override=True)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(
    config_path="./morphological_tagging/config", config_name="udify_experiment"
)
def train(config: DictConfig):
    """Train loop.

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
    print(f"\n{timer.time()} | EXPERIMENT SETUP")

    full_name = f"{config['experiment_name']}_{config['data']['language']}_{config['data']['treebank_name']}"

    full_version, experiment_dir, version = find_version(
        full_name, CHECKPOINT_DIR, debug=config["debug"]
    )

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    with open(f"{CHECKPOINT_DIR}/{full_version}/config.yaml", "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile)

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

    # == Logging
    print(f"\n{timer.time()} | LOGGER SETUP")
    if config["logging"]["logger"].lower() == "tensorboard":
        # ==== ./checkpoints/data_version/version_number

        print(f"Saving to {CHECKPOINT_DIR}/{full_version}")

        # os.path.join(save_dir, name, version)
        logger = TensorBoardLogger(
            save_dir=f"{CHECKPOINT_DIR}",
            name=f"{experiment_dir}",
            version=f"version_{version}",
            # **config["logging"]["logger_kwargs"],
        )

    elif config["logging"]["logger"].lower() in ["wandb", "weightsandbiases"]:

        logger = WandbLogger(
            project="morphological_tagging",
            save_dir=f"{CHECKPOINT_DIR}/{full_version}",
            group=f"{experiment_dir}_v{version}",
            config=OmegaConf.to_container(config),
            **config["logging"]["logger_kwargs"],
        )

    else:
        raise ConfigurationError("Logger not recognized.")

    # *==========================================================================
    # * Callbacks
    # *==========================================================================

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{CHECKPOINT_DIR}/{full_version}/checkpoints",
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["monitor_mode"],
        auto_insert_metric_name=True,
        save_last=True,
    )
    callbacks += [checkpoint_callback]

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks += [lr_monitor]

    device_monitor = DeviceStatsMonitor()
    callbacks += [device_monitor]

    prog_bar = TQDMProgressBar(refresh_rate=config["prog_bar_refresh_rate"])
    callbacks += [prog_bar]

    # *==========================================================================
    # * Dataset
    # *==========================================================================
    print(f"\n{timer.time()} | DATA SETUP")
    data_module = TreebankDataModule(**config["data"])
    data_module.prepare_data()
    data_module.setup()

    # *==========================================================================
    # * Model
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")
    corpus = data_module.corpus

    model = UDIFY(
        len_char_vocab=len(corpus.char_vocab),
        idx_char_pad=corpus.char_vocab[corpus.pad_token],
        idx_token_pad=corpus.token_vocab[corpus.pad_token],
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
        deterministic=config["deterministic"],
        fast_dev_run=(
            int(config["fdev_run"])
            if config["fdev_run"] > 0 or config["fdev_run"]
            else False
        ),
        weights_summary="top",
        **config["trainer"],
    )

    trainer.logger._default_hp_metric = None

    print(f"\n{timer.time()} | SANITY CHECK")

    trainer.validate(model, datamodule=data_module, verbose=True)

    print(f"\n{timer.time()} | TRAINING")

    trainer.fit(model, datamodule=data_module)

    # *##########
    # * TESTING #
    # *##########
    print(f"\n{timer.time()} | TESTING")
    if not (config["fdev_run"] > 0 or config["fdev_run"]):
        # If in fastdev mode, won't save a model
        # Would otherwise throw a 'PermissionError: [Errno 13] Permission denied: ...'

        print("\nTESTING")
        print(f"LOADING FROM {trainer.checkpoint_callback.best_model_path}")
        model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        model.freeze()
        model.eval()

        test_result = trainer.test(model, datamodule=data_module, verbose=True)

        timer.end()

        return test_result

    else:
        timer.end()

        return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings("ignore", message=r".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", message=r".*does not have many workers which may be a bottleneck.*"
    )
    warnings.filterwarnings("ignore", message=r".*GPU available but not used .*")
    warnings.filterwarnings("ignore", message=r".*shuffle=True")
    warnings.filterwarnings("ignore", message=r".*Trying to infer .*")
    warnings.filterwarnings(
        "ignore", message=r".*DataModule.setup has already been called.*"
    )

    train()
