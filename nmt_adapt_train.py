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
    full_version = f"{config['experiment_name']}_{config['data']['src_lang']}_{config['data']['tgt_lang']}/{run.name}"

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
    train_dataset = load_custom_dataset(
        src_lang=config["data"]["src_lang"],
        tgt_lang=config["data"]["tgt_lang"],
        dataset_name=config["data"]["dataset_name"],
        split="train",
    )
    print(f"Train dataset: {len(train_dataset)}")

    valid_dataset = load_custom_dataset(
        src_lang=config["data"]["src_lang"],
        tgt_lang=config["data"]["tgt_lang"],
        dataset_name=config["data"]["dataset_name"],
        split="test",
    )
    print(f"Valid dataset: {len(valid_dataset)}")

    # ==============================================================================
    # Index
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING INDEX" + "\n" + "+" * 50)

    if config["index"].get("fp", None) is None:
        train_index = InverseIndexv2(
            par_data=train_dataset,
            index_level=config["index"]["index_level"],
            filter_level=config["index"]["filter_level"],
        )
        print("Built index.")
        print(train_index.length_str)

    elif config["index"].get("infer_splits", False):
        train_index_fp = config["index"]["fp"] + "_train.pickle"
        train_index = InverseIndexv2.load(train_index_fp)
        print(f"Loaded index from {train_index_fp}.")

        valid_index_fp = config["index"]["fp"] + "_test.pickle"
        valid_index = InverseIndexv2.load(valid_index_fp)
        print(f"Loaded index from {valid_index_fp}.")

        print(train_index.length_str)
        print(valid_index.length_str)

    else:
        train_index_fp = config["index"]["fp"] + "_train.pickle"
        train_index = InverseIndexv2.load(train_index_fp)
        print(f"Loaded index from {train_index_fp}.")
        print(train_index.length_str)

    if config["index"].get("reduce", None) is not None:
        train_index.reduce(**config["index"]["reduce"])
        print("\nReduced index.")
        print(train_index.length_str)
        print(valid_index.length_str)

    if config["index"].get("filter", None) is not None:
        for filter_val in config["index"]["filter"]:
            train_index.filter(lambda k, f_val=filter_val: not k.contains(f_val))
            valid_index.filter(lambda k, f_val=filter_val: not k.contains(f_val))
        print(f"\nFiltered out any keys containing {config['index']['filter']}.")
        print(train_index.length_str)
        print(valid_index.length_str)

    print(f"Document coverage of train data: {train_index.coverage}/{len(train_dataset)} {train_index.coverage/len(train_dataset)*100:.2f}[%]")
    print(f"Document coverage of train data: {valid_index.coverage}/{len(valid_dataset)} {valid_index.coverage/len(valid_dataset)*100:.2f}[%]")

    # ==============================================================================
    # Task Sampler
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING TASK SAMPLER" + "\n" + "+" * 50)

    confusion_matrix = build_confusion_matrix_from_eval_data(
        fp=config["task_sampler"]["eval_fp"],
    )
    print(f"Built confusion matrix from evaluation results.")

    train_task_sampler = TaskSampler(train_index)
    train_task_sampler.set_weights(confusion_matrix)
    print(f"Built train task sampler: {train_task_sampler.length_str}")

    valid_task_sampler = TaskSampler(valid_index)
    valid_task_sampler.set_weights(confusion_matrix)
    print(f"Built valid task sampler: {valid_task_sampler.length_str}")

    # ==============================================================================
    # Trainer & Dataloader
    # ==============================================================================
    print(f"\n{timer.time()} | BUILDING TRAINER & DATALOADER" + "\n" + "+" * 50)

    config = warmup_steps_check(config)

    meta_trainer = MetaTrainer(device=device, **config["trainer"],)

    print(f"Loaded '{meta_trainer.model.name_or_path}'")
    print(f"Parameters: {round(meta_trainer.model.num_parameters()/1e+6, 1)}M")
    print(f"Device: {next(meta_trainer.model.parameters()).device}")

    meta_train_data_loader = MetaDataLoader(
        dataset=train_dataset,
        index=train_index,
        tokenizer=meta_trainer.tokenizer,
        task_sampler=train_task_sampler,
        **config["data_loader"],
    )

    meta_valid_data_loader = MetaDataLoader(
        dataset=valid_dataset,
        index=valid_index,
        tokenizer=meta_trainer.tokenizer,
        task_sampler=valid_task_sampler,
        **config["data_loader"],
    )

    print(f"\nBuilt dataloader")
    print(
        f"Max batchsize: {meta_train_data_loader.n_lemmas_per_task * meta_train_data_loader.n_samples_per_lemma}"
    )
    print(f"Probability of full NMT: {meta_train_data_loader.p_full_nmt}")
    print(f"Probability of uniform partial NMT: {meta_train_data_loader.p_uninformed}")

    wandb.watch(meta_trainer.model)

    #! #############################################################################
    #! Training
    #!
    #! This is where the magic happens
    #! #############################################################################

    if config.get("sanity_check", False):
        print(f"\n{timer.time()} | SANITY CHECK" + "\n" + "+" * 50)

        valid_logs = meta_trainer.eval_step(meta_valid_data_loader, batch_size=4, split="sanity_check")
        wandb.log(valid_logs)

    print(f"\n{timer.time()} | TRAINING" + "\n" + "+" * 50)

    best_loss = 0.0
    curr_patience = config.get("patience", config["epochs"])

    for epoch in range(config["epochs"]):

        if curr_patience <= 0:
            print("Patience ran out. Stopping early.")
            break

        print(f"\n{timer.time()} | Epoch {epoch:04}")
        for step in progressbar(
            range(config["steps_per_epoch"]), prefix=f"Epoch {epoch:03} |", size=10
        ):
            loss, logs = meta_trainer.train_step(meta_train_data_loader)

            logs = meta_trainer.optimize(loss, logs)
            logs["epoch"] = epoch
            wandb.log(logs)

        logs = meta_trainer.eval_step(meta_train_data_loader, split="train")
        logs["epoch"] = epoch
        wandb.log(logs)

        print(f"{timer.time()} | Train query losses pre-adapt {logs['train/query_harmonic_mean']:.2e}")

        logs = meta_trainer.eval_step(meta_valid_data_loader, split="valid")
        logs["epoch"] = epoch
        wandb.log(logs)

        print(f"{timer.time()} | Valid query losses pre-adapt {logs['valid/query_harmonic_mean']:.2e}")

        # Check loss if best for early stopping/saving
        # If first epoch, loss is immediately best recorded
        if epoch == 0 or logs['valid/query_harmonic_mean'] <= best_loss:
            print(f">>NEW BEST<<")
            torch.save(
                meta_trainer.model.state_dict(),
                f"{CHECKPOINT_DIR}/{full_version}/checkpoints/best.ckpt",
            )

            best_loss = logs['valid/query_harmonic_mean']

        else:
            curr_patience -= 1
            meta_trainer.meta_optimizer_scheduler.lambda_step(lambda x: x * config.get("lr_reduce", 1.0))

        # Check if should save latest
        if config.get("save_every_n", None) is not None and epoch % config.get("save_every_n") == 0:
            torch.save(
                meta_trainer.model.state_dict(),
                f"{CHECKPOINT_DIR}/{full_version}/checkpoints/latest.ckpt",
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
