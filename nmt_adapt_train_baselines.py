import os
import warnings
import yaml
from copy import deepcopy
from collections import defaultdict

# 3rd Party
import torch
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from datasets.utils.tqdm_utils import set_progress_bar_enabled

# User-defined
from nmt_adapt.data.corpus_functional import load_custom_dataset
from nmt_adapt.baselines import TokenDataloader, FineTuner, MutliTaskMorphTagTrainer
from utils.experiment import (
    find_version,
    set_seed,
    set_deterministic,
    Timer,
)
from utils.errors import ConfigurationError

CHECKPOINT_DIR = "./nmt_adapt/checkpoints"

dotenv.load_dotenv(override=True)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./nmt_adapt/config")
def train(config: DictConfig):
    """Train loop.

    """

    def warmup_steps_check(config, steps_per_epoch):
        """ Convert warmup steps defined as percentage of total training steps to a usable integer

        Args:
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        #

        cur_warmup_steps = config["trainer"]["optimizer_scheduler_kwargs"].get(
            "n_warmup_steps", False
        )
        if cur_warmup_steps and cur_warmup_steps <= 1.0 and cur_warmup_steps > 0:
            adj_n_warmup_steps = int(
                cur_warmup_steps * config["max_steps"]
            )

            print(f"Setting {adj_n_warmup_steps} as the total number of warmup steps.")
            print(f"{cur_warmup_steps*100:.2f}% of total training steps.\n")

            config["trainer"]["optimizer_scheduler_kwargs"][
                "n_warmup_steps"
            ] = adj_n_warmup_steps

        return config

    set_progress_bar_enabled(False)

    timer = Timer()

    #! #############################################################################
    #! Hyperparameters
    #!
    #! Loads in the necessary hyperparameters and checks them
    #! Probably some Hydra code
    #! #############################################################################
    if config["print_hparams"]:
        print(50 * "+")
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print(OmegaConf.to_yaml(config))
        print(50 * "+")
    else:
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print("Loaded.")

    #! #############################################################################
    #! Experiment
    #!
    #! Builds the necessary tools for experiment tracking and logging
    #! Also handles reproducibility
    #! #############################################################################
    print(f"\n{timer.time()} | EXPERIMENT SETUP")

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    torch.backends.cudnn.benchmark = False

    # == Logging
    print(f"\n{timer.time()} | LOGGER SETUP")
    if config["logging"]["logger"].lower() in ["wandb", "weightsandbiases"]:

        # TODO: figure out when fork is needed
        run = wandb.init(
            #settings=wandb.Settings(start_method="fork"),
            settings=wandb.Settings(start_method="thread"),
            config=OmegaConf.to_container(config),
            **config["logging"]["logger_kwargs"],
        )

    else:
        raise ConfigurationError("Logger not recognized.")

    # == Checkpoint dir
    full_version = f"{config['baseline']}_{config['experiment_name']}_{config['data']['src_lang']}_{config['data']['tgt_lang']}/{run.name}"

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    with open(f"{CHECKPOINT_DIR}/{full_version}/config.yaml", "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile)

    print(f"Experiment dir: {CHECKPOINT_DIR}/{full_version}")

    # == Device
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    #! #############################################################################
    #! Initialization
    #!
    #! Builds the necessary components for the training script
    #! Need to occur in specific order, but no elements should influence the state
    #! of any others, yet.
    #! #############################################################################

    # ==============================================================================
    # Dataset
    # ==============================================================================
    print(f"\n{timer.time()} | LOADING DATASET" + "\n" + "+" * 50)
    datasets = load_custom_dataset(
        src_lang=config["data"]["src_lang"],
        tgt_lang=config["data"]["tgt_lang"],
        dataset_name=config["data"]["dataset_name"],
        split="train",
    ).train_test_split(test_size=config["valid_size"])

    train_dataset = datasets["train"]
    valid_dataset = datasets["test"]

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Valid dataset: {len(valid_dataset)}")

    test_dataset = load_custom_dataset(
        src_lang=config["data"]["src_lang"],
        tgt_lang=config["data"]["tgt_lang"],
        dataset_name=config["data"]["dataset_name"],
        split="test",
    )
    print(f"Test dataset: {len(test_dataset)}")

    if config.get("cutoff", None) is not None:
        train_dataset = train_dataset.select(list(range(config["cutoff"])))
        valid_dataset = valid_dataset.select(list(range(config["cutoff"])))
        test_dataset = test_dataset.select(list(range(config["cutoff"])))

    # ==============================================================================
    # Trainer & Dataloader
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING TRAINER & DATALOADER" + "\n" + "+" * 50)

    train_dataloader = TokenDataloader(
        train_dataset,
        max_tokens=config["data_loader"]["max_tokens"],
        max_sents=config["data_loader"]["max_sents"],
        length_sort=config["data_loader"]["length_sort"],
        shuffle=True,
        )
    print(f"Train dataset batches: {len(train_dataloader)}")

    valid_dataloader = TokenDataloader(
        valid_dataset,
        max_tokens=config["data_loader"].get(
            "max_tokens_valid",
            config["data_loader"]["max_tokens"]
            ),
        max_sents=config["data_loader"]["max_sents"],
        length_sort=config["data_loader"]["length_sort"],
        shuffle=False,
        )
    print(f"Valid dataset batches: {len(valid_dataloader)}")

    test_dataloader = TokenDataloader(
        test_dataset,
        max_tokens=config["data_loader"].get(
            "max_tokens_valid",
            config["data_loader"]["max_tokens"]
            ),
        max_sents=config["data_loader"]["max_sents"],
        length_sort=config["data_loader"]["length_sort"],
        shuffle=False,
        )
    print(f"Test dataset batches: {len(test_dataloader)}")

    config = warmup_steps_check(config, len(train_dataloader))

    if config["baseline"] == "fine_tune":
        trainer = FineTuner(
            device=device,
            **config["trainer"],
        )

    elif config["baseline"] == "multi_task_morph_tag":
        tag_to_int = {tag: i for i, tag in enumerate(sorted(list({tag for tag_seq in train_dataset["morph_tags"] for tag_set in tag_seq for tag in tag_set})))}
        int_to_tag = {i: tag for tag, i in tag_to_int.items()}

        trainer = MutliTaskMorphTagTrainer(
            tag_to_int=tag_to_int,
            device=device,
            **config["trainer"],
        )

    else:
        raise ConfigurationError(f"Baseline method {config['baseline']} not implemented.")

    print(f"Loaded '{trainer.model.name_or_path}'")
    print(f"Device: {next(trainer.model.parameters()).device}")

    #! #############################################################################
    #! Training
    #!
    #! This is where the magic happens
    #! #############################################################################

    if config.get("sanity_check", False):
        print(f"\n{timer.time()} | SANITY CHECK" + "\n" + "+" * 50)

        sanity_check_dataloader = deepcopy(valid_dataloader)

        sc_batch = next(sanity_check_dataloader)
        sc_logs = trainer.eval_step(sc_batch, split="sanity_check")
        wandb.log(sc_logs)

        del sanity_check_dataloader, sc_batch, sc_logs

    print(f"\n{timer.time()} | TRAINING" + "\n" + "+" * 50)

    epoch = 0
    best_loss = 0.0
    curr_patience = config["patience"]
    for step in range(config["max_steps"]):

        if curr_patience <= 0:
            print("Patience ran out. Stopping early.")
            break

        # Train
        train_batch = next(train_dataloader)

        loss, logs = trainer.train_step(train_batch)
        logs = trainer.optimize(loss, logs)
        logs["epoch"] = epoch

        wandb.log(logs)

        # Eval
        if step % config["eval_every_n_steps"] == 0 or step == config["max_steps"] - 1:
            print(f"{timer.time()} | Epoch {epoch:04} | Validation")
            agg_eval_metrics = defaultdict(float)
            for _ in range(len(valid_dataloader)):
                valid_batch = next(valid_dataloader)
                logs = trainer.eval_step(valid_batch, split="valid")

                for k, v in logs.items():
                    agg_eval_metrics[k] += v

            agg_eval_metrics = dict(agg_eval_metrics)
            for k, v in agg_eval_metrics.items():
                if k != "valid/batch_size":
                    agg_eval_metrics[k] /= agg_eval_metrics["valid/batch_size"]

            logs["epoch"] = epoch

            wandb.log(logs)

            print(f"Validation NMT loss: {logs['valid/nmt/loss']:.2e}")

            # Check loss if best for early stopping/saving
            # If first epoch, loss is immediately best recorded
            if epoch == 0 or logs['valid/nmt/loss'] <= best_loss:
                print(f">>NEW BEST<<")
                torch.save(
                    trainer.model.state_dict(),
                    f"{CHECKPOINT_DIR}/{full_version}/checkpoints/best.ckpt",
                )
                best_loss = logs['valid/nmt/loss']
                curr_patience = config["patience"]

            else:
                curr_patience -= 1
                trainer.nmt_optimizer_scheduler.lambda_step(lambda x: x * config.get("lr_reduce", 1.0))

            # Check if should save latest
            if config.get("save_every_n", None) is not None and epoch % config.get("save_every_n") == 0:
                torch.save(
                    trainer.model.state_dict(),
                    f"{CHECKPOINT_DIR}/{full_version}/checkpoints/latest.ckpt",
                )

            epoch += 1

    timer.end()

    print(f"{timer.time()} | FINAL | Test")
    agg_eval_metrics = defaultdict(float)
    for _ in range(len(test_dataloader)):
        test_batch = next(test_dataloader)
        logs = trainer.eval_step(test_batch, split="test")

        for k, v in logs.items():
            agg_eval_metrics[k] += v

    agg_eval_metrics = dict(agg_eval_metrics)
    for k, v in agg_eval_metrics.items():
        if k != "test/batch_size":
            agg_eval_metrics[k] /= agg_eval_metrics["test/batch_size"]

    logs["epoch"] = epoch

    wandb.log(logs)

    print(f"Test NMT loss: {logs['test/nmt/loss']:.2e}")

    return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings("ignore", message=r".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", message=r".*does not have many workers which may be a bottleneck.*"
    )

    train()
