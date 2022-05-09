import os
import warnings
import yaml

# 3rd Party
import torch
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


# User-defined
from nmt_adapt.data.corpus_functional import load_custom_dataset
from nmt_adapt.inverse_index import InverseIndexv2
from nmt_adapt.task_sampling import build_confusion_matrix_from_eval_data, TaskSampler
from nmt_adapt.meta_training import MetaDataLoader, MetaTrainer
from utils.experiment import (
    find_version,
    set_seed,
    set_deterministic,
    Timer,
    progressbar,
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

    def warmup_steps_check(config):
        """ Convert warmup steps defined as percentage of total training steps to a usable integer

        Args:
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        #

        cur_warmup_steps = config["trainer"]["meta_optimizer_scheduler_kwargs"].get(
            "n_warmup_steps", False
        )
        if cur_warmup_steps and cur_warmup_steps <= 1.0 and cur_warmup_steps > 0:
            adj_n_warmup_steps = int(
                cur_warmup_steps * config["epochs"] * config["steps_per_epoch"]
            )

            print(f"Setting {adj_n_warmup_steps} as the total number of warmup steps.")
            print(f"{cur_warmup_steps*100:.2f}% of total training steps.\n")

            config["trainer"]["meta_optimizer_scheduler_kwargs"][
                "n_warmup_steps"
            ] = adj_n_warmup_steps

        return config

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

    full_name = f"{config['experiment_name']}_{config['data']['src_lang']}_{config['data']['tgt_lang']}"

    full_version, experiment_dir, version = find_version(
        full_name, CHECKPOINT_DIR, debug=config["debug"]
    )

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    with open(f"{CHECKPOINT_DIR}/{full_version}/config.yaml", "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile)

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    torch.backends.cudnn.benchmark = False

    # == Logging
    print(f"\n{timer.time()} | LOGGER SETUP")
    if config["logging"]["logger"].lower() in ["wandb", "weightsandbiases"]:

        # TODO: figure out when fork is needed
        wandb.init(
            # settings=wandb.Settings(start_method="fork"),
            **config["logging"]["logger_kwargs"],
        )

        wandb.config = config

    else:
        raise ConfigurationError("Logger not recognized.")

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
    train_dataset = load_custom_dataset(
        src_lang=config["data"]["src_lang"],
        tgt_lang=config["data"]["tgt_lang"],
        dataset_name=config["data"]["dataset_name"],
        split="train",
    )
    print(f"Train dataset: {len(train_dataset)}")
    # print(f"\Valid dataset: {len(valid_dataset)}")

    # ==============================================================================
    # Index
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING INDEX" + "\n" + "+" * 50)

    if config["index"].get("fp", None) is None:
        index = InverseIndexv2(
            par_data=train_dataset,
            index_level=config["index"]["index_level"],
            filter_level=config["index"]["filter_level"],
        )
        print("Built index.")
        print(index.length_str)

    else:
        index = InverseIndexv2.load(config["index"]["fp"])

    if config["index"].get("reduce", None) is not None:
        index.reduce(**config["index"]["reduce"])
        print(index.length_str)

    print(f"Document coverage of {index.coverage}/{len(train_dataset)}")

    # ==============================================================================
    # Task Sampler
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING TASK SAMPLER" + "\n" + "+" * 50)

    confusion_matrix = build_confusion_matrix_from_eval_data(
        fp=config["task_sampler"]["eval_fp"],
    )
    print(f"Built confusion matrix from evaluation results.")

    task_sampler = TaskSampler(index)
    task_sampler.set_weights(confusion_matrix)
    print(f"Built task sampler: {task_sampler.length_str}")

    # ==============================================================================
    # Trainer & Dataloader
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING TRAINER & DATALOADER" + "\n" + "+" * 50)

    config = warmup_steps_check(config)

    meta_trainer = MetaTrainer(device=device, **config["trainer"],)

    print(f"Loaded '{meta_trainer.model.name_or_path}'")
    print(f"Parameters: {round(meta_trainer.model.num_parameters()/1e+6, 1)}M")
    print(f"Device: {next(meta_trainer.model.parameters()).device}")

    meta_data_loader = MetaDataLoader(
        train_dataset,
        index,
        task_sampler,
        meta_trainer.tokenizer,
        **config["data_loader"],
    )

    print(f"\nBuilt dataloader")
    print(
        f"Max batchsize: {meta_data_loader.n_lemmas_per_task * meta_data_loader.n_samples_per_lemma}"
    )
    print(f"Probability of full NMT: {meta_data_loader.p_full_nmt}")
    print(f"Probability of uniform partial NMT: {meta_data_loader.p_uninformed}")

    wandb.watch(meta_trainer.model)

    #! #############################################################################
    #! Training
    #!
    #! This is where the magic happens
    #! #############################################################################

    if config.get("sanity_check", False):
        print(f"\n{timer.time()} | SANITY CHECK" + "\n" + "+" * 50)

        valid_logs = meta_trainer.eval_step(meta_data_loader, batch_size=4)
        wandb.log(valid_logs)

    print(f"\n{timer.time()} | TRAINING" + "\n" + "+" * 50)

    for epoch in range(config["epochs"]):

        for step in progressbar(
            range(config["steps_per_epoch"]), prefix=f"Epoch {epoch:03} |", size=100
        ):
            loss, train_logs = meta_trainer.train_step(meta_data_loader)

            train_logs = meta_trainer.optimize(loss, train_logs)
            wandb.log(train_logs)

        valid_logs = meta_trainer.eval_step(meta_data_loader)
        wandb.log(valid_logs)

        torch.save(
            meta_trainer.model.state_dict(),
            f"{CHECKPOINT_DIR}/{full_version}/checkpoints/model.ckpt",
        )

    timer.end()

    return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings("ignore", message=r".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", message=r".*does not have many workers which may be a bottleneck.*"
    )

    train()
