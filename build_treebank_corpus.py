import os

import hydra
from omegaconf import DictConfig

from morphological_tagging.data.corpus import TreebankDataModule

CHECKPOINT_DIR = "morphological_tagging/data/corpora"


@hydra.main(config_path="./morphological_tagging/config", config_name="treebank_corpus")
def build(config: DictConfig):
    """Builds a treebank data module from provide parameters, and saves to disk
    """

    data_module = TreebankDataModule(
        batch_size=config["batch_size"],
        language=config["language"],
        treebank_name=config["treebank_name"],
        batch_first=config["batch_first"],
        remove_unique_lemma_scripts=config["remove_unique_lemma_scripts"],
        quality_limit=config["quality_limit"],
        return_tokens_raw=config["return_tokens_raw"],
        len_sorted=config["len_sorted"],
        max_chars=config["max_chars"],
        max_tokens=config["max_tokens"],
        remove_duplicates=config["remove_duplicates"],
    )

    data_module.prepare_data()
    data_module.setup()

    output_path = os.path.join(os.getcwd(), CHECKPOINT_DIR, config["file_name"])
    print(f"Saving to: {output_path}")
    data_module.save(output_path)


if __name__ == "__main__":

    build()
