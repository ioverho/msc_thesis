import numpy as np
import datasets
from datasets import concatenate_datasets
from datasets.utils.tqdm_utils import set_progress_bar_enabled
import hydra
from omegaconf import DictConfig, OmegaConf

from nmt_adapt.data.corpus_functional import load_custom_dataset, CORPORA_LOC
from nmt_adapt.inverse_index import InverseIndexv2
from utils.experiment import set_seed, set_deterministic

CORPORA_LOC = "./nmt_adapt/data/corpora/"
INDICES_LOC = "./nmt_adapt/data/indices/"


@hydra.main(config_path="./nmt_adapt/config", config_name="merge")
def merge(config: DictConfig):

    set_progress_bar_enabled(False)

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    # Load in the separate corporas
    loaded_corpora = []
    for corpus in config["corpora"]:
        loaded_corpora.append(
            load_custom_dataset(
                config["src_lang"], config["tgt_lang"], corpus, source=None
            )
        )

    loaded_corpora_train = []
    loaded_corpora_test = []
    for corpus in loaded_corpora:
        corpus_split = corpus.train_test_split(
            test_size=config["test_size"], shuffle=True, seed=config["seed"],
        )

        loaded_corpora_train.append(corpus_split["train"])
        loaded_corpora_test.append(corpus_split["test"])

    train_dataset = concatenate_datasets(loaded_corpora_train, split="train")
    print(f"Train dataset: {len(train_dataset)}")

    train_dataset = train_dataset.filter(
        lambda example, idx: len(example["tgt_tokens"]) > 0 and len(example["src_text"]) > 0,
            with_indices=True
            )
    print(f"Without empty strings: {len(train_dataset)}")

    delta_lens = [
        abs(len(tgt_tokens) - len(src_text.split(" "))) / len(src_text.split(" "))
        for tgt_tokens, src_text in zip(train_dataset["tgt_tokens"], train_dataset["src_text"])]

    _, max_diff = np.quantile(delta_lens, [0.001, 0.999])

    train_dataset = train_dataset.filter(
        lambda example, idx: abs(len(example["tgt_tokens"]) - len(example["src_text"].split(" "))) \
            / len(example["src_text"].split(" ")) < max_diff,
            with_indices=True
            )
    print(f"Removing sentences with relative lengths larger than {max_diff}")
    print(f"Filtered train dataset: {len(train_dataset)}")

    test_dataset = concatenate_datasets(loaded_corpora_test, split="test")
    print(f"\nTest dataset: {len(test_dataset)}")

    test_dataset = test_dataset.filter(
        lambda example, idx: len(example["tgt_tokens"]) > 0 and len(example["src_text"]) > 0,
            with_indices=True
            )
    print(f"Without empty strings: {len(test_dataset)}")

    delta_lens = [
        abs(len(tgt_tokens) - len(src_text.split(" "))) / len(src_text.split(" "))
        for tgt_tokens, src_text in zip(test_dataset["tgt_tokens"], test_dataset["src_text"])]

    min_diff, max_diff = np.quantile(delta_lens, [0.001, 0.999])

    test_dataset = test_dataset.filter(
        lambda example, idx: abs(len(example["tgt_tokens"]) - len(example["src_text"].split(" "))) \
            / len(example["src_text"].split(" ")) < max_diff,
            with_indices=True
            )
    print(f"Removing sentences with relative lengths larger than {max_diff}")
    print(f"Filtered test dataset: {len(test_dataset)}")

    train_dataset.save_to_disk(
        CORPORA_LOC
        + config["agg_name"]
        + f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}_train"
    )

    test_dataset.save_to_disk(
        CORPORA_LOC
        + config["agg_name"]
        + f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}_test"
    )

    # ==========================================================================
    # Train index
    # ==========================================================================
    print("\nBuilding train index.")
    train_index = InverseIndexv2(
        par_data=train_dataset,
        index_level=config["index"]["index_level"],
        filter_level=config["index"]["filter_level"],
    )
    print("Built index.")
    print(train_index.length_str)
    train_index.reduce(**config["index"]["reduce"])
    print(train_index.length_str)
    print(f"Document coverage of {train_index.coverage}/{len(train_dataset)}")

    train_index.save(
        INDICES_LOC
        + config["agg_name"]
        + f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}_train.pickle"
    )

    # ==========================================================================
    # Test index
    # ==========================================================================
    print("\nBuilding test index.")
    test_index = InverseIndexv2(
        par_data=test_dataset,
        index_level=config["index"]["index_level"],
        filter_level=config["index"]["filter_level"],
    )
    print("Built index.")
    print(test_index.length_str)
    test_index.reduce(**config["index"]["reduce"])
    print(test_index.length_str)
    print(f"Document coverage of {test_index.coverage}/{len(test_dataset)}")

    test_index.save(
        INDICES_LOC
        + config["agg_name"]
        + f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}_test.pickle"
    )


if __name__ == "__main__":

    merge()
