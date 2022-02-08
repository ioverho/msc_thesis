import os
import argparse
import warnings
from shutil import copyfile
import yaml
import math

import dotenv
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profiler

from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.models.udpipe2 import UDPipe2
from morphological_tagging.metrics import RunningStats
from utils.experiment import (
    set_seed,
    set_deterministic,
    Timer,
    progressbar,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CHECKPOINT_DIR = "./morphological_tagging/checkpoints"

dotenv.load_dotenv(override=True)


class ModelCheckpoint:
    def __init__(
        self,
        monitor: str,
        monitor_mode: str,
        save_dir: str = None,
        save_last: bool = True,
        verbose: bool = True,
    ):

        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.save_dir = save_dir if save_dir is not None else CHECKPOINT_DIR
        self.save_last = save_last
        self.verbose = verbose

        self.best_path = None
        self.last_path = None

        self._metric_values = RunningStats()
        self._i = 0

    def _check(self, val):

        if self._i > 0 and self.monitor_mode == "max":
            if val >= self._metric_values.min:
                return True, self._metric_values.max
            else:
                return False, self._metric_values.max

        elif self._i > 0 and self.monitor_mode == "min":
            if val <= self._metric_values.min:
                return True, self._metric_values.min
            else:
                return False, self._metric_values.min

        else:
            return True, math.nan

    def __call__(self, model, metrics):

        cur_val = metrics[self.monitor]

        checkpoint_bool, prev_val = self._check(cur_val)

        if checkpoint_bool:
            if self.best_path is not None:
                os.remove(self.best_path)

            self.best_path = (
                f"{self.save_dir}/best_{self._i}_{self.monitor}_{cur_val:.2e}.pt"
            )
            torch.save(
                model.state_dict(), self.best_path,
            )

            if self.verbose:
                print(f">> New best model {self.monitor}={cur_val:.2e} <<")
                print(
                    f">> Change: {cur_val-prev_val:+.2e}, {(cur_val-prev_val)/prev_val*100:+.2f}% <<"
                )
        elif self.verbose:
            print(f"\nNo recorded improvement.")

        if self.save_last:
            if self.last_path is not None:
                os.remove(self.last_path)

            self.last_path = (
                f"{self.save_dir}/last_{self._i}_{self.monitor}_{cur_val:.2e}.pt"
            )

            torch.save(
                model.state_dict(), self.last_path,
            )

        self._metric_values(cur_val)
        self._i += 1


@hydra.main(
    config_path="./morphological_tagging/config", config_name="udpipe2_experiment"
)
def train(config: DictConfig) -> None:
    def make_model(config):

        model = UDPipe2(
            char_vocab=data_module.corpus.char_vocab,
            token_vocab=data_module.corpus.token_vocab,
            unk_token=data_module.corpus.unk_token,
            pad_token=data_module.corpus.pad_token,
            pretrained_embedding_dim=data_module.corpus.pretrained_embeddings_dim,
            n_lemma_scripts=len(data_module.corpus.script_counter),
            n_morph_tags=len(data_module.corpus.morph_tag_vocab),
            n_morph_cats=len(data_module.corpus.morph_cat_vocab),
            preprocessor_kwargs=config.preprocessor,
            **config["model"],
        )

        model = model.to(device)

        return model

    def log(split: str, metrics, epoch: int, **log_kwargs):

        print(
            f"Loss:  {metrics[f'{split}_loss_total']:5.2e} ({metrics[f'{split}_loss_lemma']:5.2e}/{metrics[f'{split}_loss_morph']:5.2e}/{metrics[f'{split}_loss_morph_reg']:5.2e})"
        )
        print(
            f"Lemma: {metrics[f'{split}_lemma_acc']*100:5.2f} ({metrics[f'{split}_lemma_acc_se']:5.2e}), {metrics[f'{split}_lemma_dist']:.2f} ({0:5.2e})"
        )
        print(
            f"Morph: {metrics[f'{split}_morph_set_acc']*100:5.2f} ({metrics[f'{split}_morph_set_acc_se']:5.2e}), {metrics[f'{split}_morph_f1']*100:5.2f} ({metrics[f'{split}_morph_precision']*100:5.2f}/{metrics[f'{split}_morph_recall']*100:5.2f})"
        )

        metrics["epoch"] = epoch
        wandb.log(metrics, **log_kwargs)
        model.clear_metrics(f"{split}")
        print("")

    def train_epoch(dataloader, split: str = "train", scheduler_freq: str = "epoch"):

        model.train()

        for batch in progressbar(
            dataloader, prefix=f"{split}", size=config["prog_bar_size"]
        ):

            loss = model.step(batch, split=f"{split}")

            loss.backward()

            clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            for opt in model.optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            if scheduler_freq == "step":
                # Step schedulers
                for sch in model.schedulers:
                    sch.step()

        if scheduler_freq == "epoch":
            # Step schedulers
            for sch in model.schedulers:
                sch.step()

        log(f"{split}", model.log_metrics(f"{split}"), epoch, commit=False)

    def eval_epoch(
        dataloader,
        split: str = "train",
        checkpoint_check: bool = False,
        epoch_shift: int = 0,
    ):

        model.eval()

        for batch in progressbar(
            dataloader, prefix=f"{split}", size=config["prog_bar_size"]
        ):
            with torch.no_grad():
                model.step(batch, split=f"{split}")

        eval_metrics = model.log_metrics(f"{split}")

        log(f"{split}", eval_metrics, epoch + epoch_shift)

        if checkpoint_check:
            checkpoint_callback(model, eval_metrics)

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

    # == Logger
    experiment = wandb.init(
        config=OmegaConf.to_container(config),
        dir=f"{CHECKPOINT_DIR}/{full_name}",
        **config["logger"],
    )

    # == Version
    # ==== ./checkpoints/data_version/version_number
    local_save_dir = f"{CHECKPOINT_DIR}/{full_name}/{experiment.name}"
    os.makedirs(local_save_dir)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_name}/wandb", exist_ok=True)

    with open(f"{local_save_dir}/config.yaml", "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile)

    print(f"Saving to {local_save_dir}")

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
    data_module = TreebankDataModule(**config["data"])
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    valid_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # *==========================================================================
    # * MODEL INITIALIZATION
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")
    model = make_model(config)

    checkpoint_callback = ModelCheckpoint(
        monitor=config["monitor"],
        monitor_mode=config["monitor_mode"],
        save_dir=local_save_dir,
        save_last=True,
    )

    # *==========================================================================
    # * TRAINING
    # *==========================================================================
    print(f"\n{timer.time()} | TRAINING")

    for epoch in range(0, config["max_epochs"] + 1):
        # Full epoch ===========================================================
        print(
            f"\n{timer.time()} | EPOCH {epoch:0{len(str(config['max_epochs']))}d}/{config['max_epochs']:d}"
        )

        if epoch != 0:
            # Skip the first epoch.
            # Also ensures exactly `max_epochs` epochs are run

            # Train epoch ==========================================================
            train_epoch(train_loader, split="train", scheduler_freq="epoch")

            # Validation epoch =====================================================
            eval_epoch(valid_loader, split="valid", checkpoint_check=True)

        else:
            # First epoch. Only run a validation epoch on test set for checking initial loss values
            eval_epoch(test_loader, split="test", checkpoint_check=False)

    # *==========================================================================
    # * TESTING
    # *==========================================================================
    print(f"\n{timer.time()} | TESTING")
    model = make_model(config)

    model.load_state_dict(
        torch.load(checkpoint_callback.best_path, map_location=device)
    )
    model.eval()

    eval_epoch(train_loader, split="train", epoch_shift=1, checkpoint_check=False)

    eval_epoch(valid_loader, split="valid", epoch_shift=1, checkpoint_check=False)

    eval_epoch(test_loader, split="test", epoch_shift=1, checkpoint_check=False)

    timer.end()

    return 1


if __name__ == "__main__":
    train()
