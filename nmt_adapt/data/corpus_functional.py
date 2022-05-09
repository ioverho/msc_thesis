import os
import urllib.request
import certifi
import ssl
import zipfile
import jsonlines as jsonl
import typing
from collections import defaultdict

import pycountry
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from datasets.fingerprint import set_caching_enabled

from utils.errors import ConfigurationError
from utils.tokenizers import MosesTokenizerWrapped

RAW_LOC = "./nmt_adapt/data/raw"
CORPORA_LOC = "./nmt_adapt/data/corpora"

OPUS_URLS = {
    "EUbookshop": lambda x, y: f"https://opus.nlpl.eu/download.php?f=EUbookshop/v2/moses/{x}-{y}.txt.zip",  # EU books
    "GlobalVoices": lambda x, y: f"https://opus.nlpl.eu/download.php?f=GlobalVoices/v2018q4/moses/{x}-{y}.txt.zip",  # News stories
    "GNOME": lambda x, y: f"https://opus.nlpl.eu/download.php?f=GNOME/v1/moses/{x}-{y}.txt.zip",  # Software localization
    "Ubuntu": lambda x, y: f"https://opus.nlpl.eu/download.php?f=Ubuntu/v14.10/moses/{x}-{y}.txt.zip",  # Software localization
    "KDE4": lambda x, y: f"https://opus.nlpl.eu/download.php?f=KDE4/v2/moses/{x}-{y}.txt.zip",  # Software localization
    "OpenSubtitles": lambda x, y: f"https://opus.nlpl.eu/download.php?f=OpenSubtitles/v1/moses/{x}-{y}.txt.zip",  # Subtitles
    "TED2020": lambda x, y: f"https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/{x}-{y}.txt.zip",  # TED talks
}

set_caching_enabled(False)


def load_custom_dataset(
    src_lang: str,
    tgt_lang: str,
    dataset_name: str,
    source: typing.Optional[str] = None,
    split: typing.Optional[str] = None,
    filter_unique_texts: bool = False,
):
    """Loads in a parallel dataset for annotation and such.

    Args:
        src_lang (str): the source language, full name
        tgt_lang (str): the target language, full name
        dataset_name (str): the name of the dataset. Choice of {"flores", "{hf_name}", "{opus_name}"}
        source (typing.Optional[str], optional): the location where one should look for the dataset. If "hf", will download from HF hub, if "opus" wil download from opus.nlp.eu, if `None` assumes processed file already in `CORPORA_LOC`. Defaults to None.
        split (typing.Optional[str], optional): the dataset split to load in. Defaults to None.
        filter_unique_texts (bool, optional): whether or not to filter texts (target side). Only recommneded for smaller datasets. Defaults to False.

    Returns:
        datasets.Dataset: a HF compatible dataset
    """

    src = pycountry.languages.get(name=src_lang)
    tgt = pycountry.languages.get(name=tgt_lang)

    tokenizer = MosesTokenizerWrapped(lang=tgt.name)

    dataset_name_lower = dataset_name.lower()
    dataset_dir = f"{dataset_name_lower}_{src_lang.lower()}_{tgt_lang.lower()}"
    if split is not None:
        dataset_dir += f"_{split}"

    if source is None:
        if os.path.isdir(f"{CORPORA_LOC}/{dataset_dir}"):
            print(f"Loading pre-built dataset from: {CORPORA_LOC}/{dataset_dir}")
            return Dataset.load_from_disk(f"{CORPORA_LOC}/{dataset_dir}")

        else:
            raise ConfigurationError(f"{dataset_dir} not found in {CORPORA_LOC}")

    elif source.lower() == "hf":
        try:
            dataset = load_dataset(
                dataset_name_lower, lang1=src.alpha_2, lang2=tgt.alpha_2, split=split
            )
        except FileNotFoundError:
            dataset = load_dataset(
                dataset_name_lower, lang1=tgt.alpha_2, lang2=src.alpha_2, split=split
            )

    elif source == "opus":

        if os.path.isfile(f"{RAW_LOC}/{dataset_dir}/bitext.jsonl"):
            print(f"Dataset {dataset_dir} already exists on drive.")
        else:
            os.makedirs(f"{RAW_LOC}/{dataset_dir}", exist_ok=True)

            # Sort the languages alphabetically
            l1, l2 = sorted([src.alpha_2, tgt.alpha_2])

            # Download from OPUS
            url = OPUS_URLS[dataset_name](l1, l2)
            print(f"Downloading from {url}")

            with urllib.request.urlopen(
                url, context=ssl.create_default_context(cafile=certifi.where())
            ) as response, open(
                f"{RAW_LOC}/{dataset_dir}/download.zip", "wb"
            ) as out_file:
                data = response.read()
                out_file.write(data)

            # Extract the zip file
            with zipfile.ZipFile(
                f"{RAW_LOC}/{dataset_dir}/download.zip", "r"
            ) as zip_ref:
                zip_ref.extractall(f"{RAW_LOC}/{dataset_dir}")

            # Transition to json file
            # Memory efficient
            src_obj = open(
                f"./nmt_adapt/data/raw/{dataset_dir}/{dataset_name}.{l1}-{l2}.{src.alpha_2}",
                "r",
                encoding="utf-8",
            )
            tgt_obj = open(
                f"./nmt_adapt/data/raw/{dataset_dir}/{dataset_name}.{l1}-{l2}.{tgt.alpha_2}",
                "r",
                encoding="utf-8",
            )
            writer = jsonl.open(f"./nmt_adapt/data/raw/{dataset_dir}/bitext.jsonl", "w")

            for i, (src_text, tgt_text) in enumerate(zip(src_obj, tgt_obj)):
                writer.write(
                    {
                        "id": f"{dataset_name_lower}_{i}",
                        "translation": {
                            src.alpha_2: src_text.strip(),
                            tgt.alpha_2: tgt_text.strip(),
                        },
                    }
                )

            src_obj.close()
            tgt_obj.close()
            writer.close()

        # Load dataset using json file
        dataset = Dataset.from_json(f"./nmt_adapt/data/raw/{dataset_dir}/bitext.jsonl")

    elif dataset_name_lower == "flores":

        if split == "dev" or split == "valid":
            splits = ["dev"]
        elif split == "devtest" or split == "test":
            splits = ["devtest"]
        elif split == "both" or split == "all":
            splits = ["dev", "devtest"]
        else:
            raise ValueError(f"Split {split} not recognized.")

        all_texts = {"id": [], "translation": []}
        i = 0
        for spl in splits:
            texts = []
            for lang in [src.alpha_3, tgt.alpha_3]:
                with open(
                    f"{RAW_LOC}/flores101_dataset/{spl}/{lang}.{spl}", "rb"
                ) as fp:
                    flores_text = fp.readlines()

                texts.append([f_text.strip().decode("utf-8") for f_text in flores_text])

            for ii, (src_text, tgt_text), in enumerate(zip(*texts)):
                all_texts["id"].append(i)
                all_texts["translation"].append(
                    {src.alpha_2: src_text, tgt.alpha_2: tgt_text}
                )

                i += 1

        dataset = Dataset.from_dict(all_texts, split="test")

    else:
        raise ConfigurationError(f"{source} / {dataset_name_lower} not implemented.")

    dataset = dataset.map(
        lambda x: {
            "src_text": x["translation"][src.alpha_2],
            "tgt_text": x["translation"][tgt.alpha_2],
            "tgt_tokens": tokenizer(x["translation"][tgt.alpha_2]),
        }
    )

    if filter_unique_texts:

        pre_n_rows = dataset.num_rows

        index = defaultdict(list)
        for i, txt in zip(dataset["id"], dataset["tgt_text"]):
            index[txt].append(i)

        keys_to_keep = {v[0] for v in index.values()}

        dataset = dataset.filter(lambda x: x["id"] in keys_to_keep)

        post_n_rows = dataset.num_rows

        print(f"Went from {pre_n_rows} to {post_n_rows} examples.")

    dataset = dataset.remove_columns(["translation", "tgt_text"])
    dataset.info.config_name = dataset_dir

    return dataset
