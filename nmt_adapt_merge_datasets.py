import datasets
from datasets import concatenate_datasets
import hydra
from omegaconf import DictConfig, OmegaConf

from nmt_adapt.data.corpus_functional import load_custom_dataset, CORPORA_LOC
from nmt_adapt.inverse_index import InverseIndexv2
from utils.experiment import set_seed, set_deterministic

CORPORA_LOC = "./nmt_adapt/data/corpora/"
INDICES_LOC = "./nmt_adapt/data/indices/"


@hydra.main(config_path="./nmt_adapt/config", config_name="merge")
def merge(config: DictConfig):

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
    test_dataset = concatenate_datasets(loaded_corpora_test, split="test")

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
