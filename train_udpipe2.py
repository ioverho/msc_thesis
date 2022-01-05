# *==============================================================================
# *Package import
# *==============================================================================
import os
import yaml
import argparse
import warnings
from shutil import copyfile

# 3rd Party
from tqdm import tqdm
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

# User-defined
from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.models.udpipe2 import UDPipe2
from morphological_tagging.models.preprocessor import UDPipe2PreProcessor
from utils.experiment import find_version, set_seed, set_deterministic, Timer
from utils.errors import ConfigurationError

CHECKPOINT_DIR = "./morphological_tagging/checkpoints"

dotenv.load_dotenv(override=True)

def train(args):
    """Train loop.

    """

    timer = Timer()

    # *==========================================================================
    # *Config reading
    # *==========================================================================
    print(f"Config path: {args.config_file_path}\n\n")

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

    full_name = f"{config['run']['experiment_name']}_{config['data']['language']}_{config['data']['treebank_name']}"

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
    data_module = TreebankDataModule(**config['data'])
    data_module.prepare_data()
    data_module.setup()

    print(f"\n{timer.time()} | DATA PRE-PROCESSING")
    preprocessor = UDPipe2PreProcessor(
        language=config['data']['language'],
        **config['preprocess']
    )

    for batch in tqdm(data_module._preprocess_dataloader(
        batch_size= preprocessor.batch_size if preprocessor.batch_size else config['data']['batch_size']
        ), mininterval=5):
        preprocessor(batch, set_doc_attr=True)

    # *==========================================================================
    # * Model
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")
    model = UDPipe2(
        char_vocab=data_module.corpus.char_vocab,
        token_vocab=data_module.corpus.token_vocab,
        unk_token=data_module.corpus.unk_token,
        pad_token=data_module.corpus.pad_token,
        pretrained_embedding_dim=data_module.corpus.pretrained_embeddings_dim,
        n_lemma_scripts=len(data_module.corpus.script_counter),
        n_morph_tags=len(data_module.corpus.morph_tag_vocab),
        n_morph_cats=len(data_module.corpus.morph_cat_vocab),
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

    print(f"\n{timer.time()} | SANITY CHECK")

    trainer.test(model, datamodule=data_module, verbose=True)

    print(f"\n{timer.time()} | TRAINING")

    trainer.fit(model, datamodule=data_module)

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

        test_result = trainer.test(model, datamodule=data_module, verbose=True)

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
    warnings.filterwarnings("ignore", message=r".*DataModule.setup has already been called.*")

    train(args)
