
#! DEPRECATED
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import requests as req
from typing import Optional, List
import shutil
import urllib.request
import tarfile
import gzip
from collections import defaultdict
import typing
import json

from datasets import load_dataset
import pycountry
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from datasets.fingerprint import set_caching_enabled

from utils.errors import ConfigurationError
from utils.tokenizers import MosesTokenizerWrapped

CORPORA_LOC = "./nmt_adapt/data/corpora/"

SOURCE_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/OPUS-MT-testsets/master/testsets/"
)

FLORES_LOC = Path("./nmt_adapt/data/corpora/flores101_dataset")

TATOEBA_TEST_VERSIONS = [
    "tatoeba-test-v2020-07-28",
    "tatoeba-test-v2021-03-30",
    "tatoeba-test-v2021-08-07",
]

DUMP_LOC = Path("./nmt_adapt/data/corpora/tatoeba")

TATOEBA_TRAIN_DUMP_LOC = "./nmt_adapt/data/corpora/tatoeba_train"


@dataclass
class AnnotatedSentence:

    source_file: Optional[str] = None
    id: Optional[str] = None
    split: Optional[str] = None
    parallel_text: Optional[str] = None
    tokens: Optional[List[str]] = None
    lemmas: Optional[List[str]] = None
    lemma_scripts: Optional[List[str]] = None
    morph_tags: Optional[List[str]] = None
    morph_cats: Optional[List[str]] = None

    def __str__(self):
        return f"Doc(id={self.id})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.tokens)

    def to_json(self):

        state_dict = dict()

        if self.source_file is not None:
            state_dict.update({"source_file": self.source_file})
        if self.id is not None:
            state_dict.update({"id": self.id})
        if self.split is not None:
            state_dict.update({"split": self.split})
        if self.parallel_text is not None:
            state_dict.update({"parallel_text": self.parallel_text})
        if self.tokens is not None:
            state_dict.update({"tokens": self.tokens})
        if self.lemmas is not None:
            state_dict.update({"lemmas": self.lemmas})
        if self.lemma_scripts is not None:
            state_dict.update({"lemma_scripts": self.lemma_scripts})
        if self.morph_tags is not None:
            state_dict.update({"morph_tags": self.morph_tags})
        if self.morph_cats is not None:
            state_dict.update({"morph_cats": self.morph_cats})

        return state_dict


class ParallelTreebankCorpus(Dataset):
    def __init__(
        self, src_lang: str, tgt_lang: str, tokenizer: Optional[callable] = None
    ):

        self.src_lang = pycountry.languages.get(name=src_lang)
        self.tgt_lang = pycountry.languages.get(name=tgt_lang)

        self.tokenizer = tokenizer

        self.sents = []
        self._included_texts = set()
        self.splits = defaultdict(list)

        self.parallel_dataset = []

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i: int):
        return self.sents[i]

    @property
    def src_lang_full(self):
        return self.src_lang.name.lower()

    @property
    def tgt_lang_full(self):
        return self.tgt_lang.name.lower()

    def load_tatoeba_train_dataset(self):

        if os.path.isdir(
            f"{TATOEBA_TRAIN_DUMP_LOC}/{self.src_lang.alpha_3}-{self.tgt_lang.alpha_3}"
        ):
            lang_order = 0
            f_name = f"{self.src_lang.alpha_3}-{self.tgt_lang.alpha_3}"
            print("Skipping download.")

        elif os.path.isdir(
            f"{TATOEBA_TRAIN_DUMP_LOC}/{self.tgt_lang.alpha_3}-{self.src_lang.alpha_3}"
        ):
            lang_order = 1
            f_name = f"{self.tgt_lang.alpha_3}-{self.src_lang.alpha_3}"
            print("Skipping download.")

        else:
            # Import based on correct ordering
            try:
                urllib.request.urlretrieve(
                    f"https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/{self.src_lang.alpha_3}-{self.tgt_lang.alpha_3}.tar",
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{self.src_lang.alpha_3}-{self.tgt_lang.alpha_3}.tar",
                )
                lang_order = 0
                f_name = f"{self.src_lang.alpha_3}-{self.tgt_lang.alpha_3}"
            except:
                urllib.request.urlretrieve(
                    f"https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/{self.tgt_lang.alpha_3}-{self.src_lang.alpha_3}.tar",
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{self.tgt_lang.alpha_3}-{self.src_lang.alpha_3}.tar",
                )
                lang_order = 1
                f_name = f"{self.tgt_lang.alpha_3}-{self.src_lang.alpha_3}"

            archive_loc = f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}.tar"

            # Extract files
            with tarfile.open(archive_loc, "r:") as f:
                f.extractall(f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/")

            for parent, sub_dirs, sub_files in os.walk(
                f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/"
            ):
                if len(sub_dirs) == 0:
                    for f in sub_files:
                        if ".gz" in f:
                            with gzip.open(f"{parent}/{f}", "rb") as f_in:
                                with open(
                                    f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{f[:-3]}", "wb"
                                ) as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                        else:
                            shutil.move(
                                parent + "/" + f,
                                f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{f}",
                            )

            # Clean-up unneeded files and directories
            shutil.rmtree(f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/data/")
            os.remove(archive_loc)

        # Check for individual splits
        split_files = defaultdict(list)

        for parent, sub_dirs, sub_files in os.walk(
            f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}"
        ):
            for f in sub_files:
                split = f.split(".")[0]
                if split == "README":
                    continue

                split_files[split].append(f)

        split_files = dict(split_files)

        for split in split_files.keys():
            id_obj = open(f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{split}.id", "r")
            if not lang_order:
                src_obj = open(
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{split}.src",
                    "r",
                    encoding="utf8",
                )
                tgt_obj = open(
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{split}.trg",
                    "r",
                    encoding="utf8",
                )
            else:
                src_obj = open(
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{split}.trg",
                    "r",
                    encoding="utf8",
                )
                tgt_obj = open(
                    f"{TATOEBA_TRAIN_DUMP_LOC}/{f_name}/{split}.src",
                    "r",
                    encoding="utf8",
                )

            for doc_id, src_text, tgt_text in zip(id_obj, src_obj, tgt_obj):
                doc_id = doc_id.strip().replace("\t", "_")
                src_text = src_text.strip()
                tgt_text = tgt_text.strip()

                self.splits[split].append(len(self.sents))
                self.sents.append(
                    AnnotatedSentence(
                        source_file="tatoeba",
                        id=doc_id,
                        parallel_text=src_text,
                        tokens=self.tokenizer(tgt_text),
                    )
                )

        id_obj.close()
        src_obj.close()
        tgt_obj.close()

    def load_tatoeba_dataset(self):

        if (DUMP_LOC / f"tatoeba_test_{self.src_lang}_{self.tgt_lang}.pickle").exists():
            print("Dataset found.")
            with open(
                DUMP_LOC / f"tatoeba_test_{self.src_lang}_{self.tgt_lang}.pickle", "rb"
            ) as fp:
                all_texts = pickle.load(fp)

        else:
            print("Dataset not found.\nDownloading.")
            src_3 = pycountry.languages.get(alpha_2=self.src_lang).alpha_3
            tgt_3 = pycountry.languages.get(alpha_2=self.tgt_lang).alpha_3

            all_texts = []

            i = 0
            with open(
                DUMP_LOC / f"tatoeba_test_{self.src_lang}_{self.tgt_lang}.pickle", "wb"
            ) as fp:
                for version in TATOEBA_TEST_VERSIONS:
                    texts = []
                    for lang in [src_3, tgt_3]:
                        file_url = f"{SOURCE_URL}/{src_3}-{tgt_3}/{version}.{lang}"

                        res = req.get(file_url)

                        text = res.text.split("\n")

                        if len(text) == 1:
                            file_url = f"{SOURCE_URL}/{tgt_3}-{src_3}/{version}.{lang}"

                            res = req.get(file_url)

                            text = res.text.split("\n")

                        texts.append(text)

                    for ii, (src_text, tgt_text), in enumerate(zip(*texts)):
                        all_texts.append(
                            {
                                "id": i,
                                "source": f"{version}_{ii}",
                                self.src_lang: src_text,
                                self.tgt_lang: tgt_text,
                            }
                        )
                        i += 1

                pickle.dump(all_texts, fp)

        self.parallel_dataset.extend(all_texts)

    def load_flores_dataset(self, split: str):

        src_3 = pycountry.languages.get(alpha_2=self.src_lang).alpha_3
        tgt_3 = pycountry.languages.get(alpha_2=self.tgt_lang).alpha_3

        if split == "dev" or split == "valid":
            splits = ["dev"]
        elif split == "devtest" or split == "test":
            splits = ["devtest"]
        elif split == "both" or split == "all":
            splits = ["dev", "devtest"]

        all_texts = []
        i = 0
        for spl in splits:
            texts = []
            for lang in [src_3, tgt_3]:
                with open(FLORES_LOC / spl / f"{lang}.{spl}", "rb") as fp:
                    flores_text = fp.readlines()

                texts.append([f_text.strip().decode("utf-8") for f_text in flores_text])

            for ii, (src_text, tgt_text), in enumerate(zip(*texts)):
                all_texts.append(
                    {
                        "id": i,
                        "source": f"flores101_{spl}_{ii}",
                        self.src_lang: src_text,
                        self.tgt_lang: tgt_text,
                    }
                )
                i += 1

        self.parallel_dataset.extend(all_texts)

    def load_hf_dataset(self, path: str, name: str, **load_dataset_kwargs):
        if (self.src_lang not in name) and (self.tgt_lang not in name):
            raise ConfigurationError(
                "Given source and target language not in the dataset name."
            )

        self.parallel_dataset_path = path
        self._parallel_dataset_kwargs = {
            "path": path,
            "name": name,
            **load_dataset_kwargs,
        }

        self.parallel_dataset.extend(
            [
                {
                    "id": sample["id"],
                    "source": path,
                    self.src_lang: sample["translation"][self.src_lang],
                    self.tgt_lang: sample["translation"][self.tgt_lang],
                }
                for sample in load_dataset(**self._parallel_dataset_kwargs)
            ]
        )

    def extend(self, sequence_of_annotated_sents):

        for annotated_sent in sequence_of_annotated_sents:
            tgt_text = " ".join(annotated_sent.tokens)
            if tgt_text in self._included_texts:
                continue

            self.sents.append(annotated_sent)
            self._included_texts.add(tgt_text)

    def save(self, fp):

        del self.parallel_dataset
        del self._included_texts

        with open(fp, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, fp):

        with open(fp, "rb") as fp:
            return pickle.load(fp)


class EfficientParallelTreebankCorpus(ParallelTreebankCorpus):
    """_summary_

    Args:
        ParallelTreebankCorpus (_type_): _description_
    """

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        tokenizer: typing.Optional[callable] = None,
        loc: typing.Optional[str] = None,
    ):
        super().__init__(src_lang, tgt_lang, tokenizer)

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self.loc = f"{CORPORA_LOC}/{loc}"

        os.makedirs(self.loc, exist_ok=True)

        self.dataset = None
        self._intermediate_files = []

    def shard_dataset(self):

        dataset_json = {k: {"data": []} for k in self.splits.keys()}
        for split, split_ids in self.splits.items():
            for doc_id in split_ids:
                dataset_json[split]["data"].append(self.sents[doc_id].to_json())

        for split in dataset_json.keys():
            with open(f"{self.loc}/dataset_{split}.json", "w") as f:
                json.dump(dataset_json[split], f)
                self._intermediate_files.append(f)

        del self.sents
        del self._included_texts
        del self.splits
        del self.parallel_dataset

    def load_dataset(self, is_pre_formatted: bool = False, **kwargs):

        if not is_pre_formatted:
            self.dataset = load_dataset(
                name="json", path=self.loc, field="data", **kwargs
            )
        else:
            self.dataset = DatasetDict.from_disk(self.loc, **kwargs)

    def save(self, fp: typing.Optional[str] = None):

        if fp is None:
            fp = self.loc + "/state_dict.pickle"

        with open(fp, "wb") as fp:
            pickle.dump(
                {
                    "src_lang": self._src_lang,
                    "tgt_lang": self._tgt_lang,
                    "tokenizer": self.tokenizer,
                    "loc": self.loc,
                    "_intermediate_files": self._intermediate_files,
                },
                fp,
            )

        if self.dataset is not None:
            self.dataset.save_to_disk(self.loc)

    @classmethod
    def load(cls, fp, **kwargs):

        state_dict = pickle.load(fp)

        corpus = EfficientParallelTreebankCorpus(
            state_dict["src_lang"],
            state_dict["tgt_lang"],
            state_dict["tokenizer"],
            state_dict["loc"],
        )

        corpus._intermediate_files = state_dict["_intermediate_files"]

        corpus.load_dataset(**kwargs)

        with open(fp, "rb") as fp:
            return corpus

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]

    def clean_up(self):
        for f in self._intermediate_files:
            os.remove(f)
