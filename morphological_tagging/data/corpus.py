import os
import csv
import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import torchtext
from torchtext.vocab import build_vocab_from_iterator, Vectors
import pytorch_lightning as pl
import pandas as pd

from morphological_tagging.data.lemma_script import LemmaScriptGenerator


BASEPATH = "./morphological_tagging/data/sigmorphon_2019/task2"

ALL_TAGS_FP = "./morphological_tagging/data/uni_morph_tags.json"


def get_conllu_files(language: str, name: str = "merge") -> Union[dict[list], list]:
    """Get the UD/CONLLU/SIGMORPHON treebanks from the BASEPATH dir.

    Args:
        language (str): the desired language
        name (str): the desired treebank name, or merge for all. Default to "merge"

    Returns:
        Union[dict[list], list]: a dict, with as keys the data split and as values a list with filepaths.
            If `splits` is False, then return a list.
    """

    files = defaultdict(list)
    for (dirpath, _, filenames) in os.walk(BASEPATH):
        if len(filenames) == 0:
            continue

        treebank = os.path.split(dirpath)[-1]
        _, treebank = re.split(r"_", treebank, maxsplit=1)
        t_lang, t_name = re.split(r"-", treebank, maxsplit=1)

        if t_lang.lower() == language.lower() and (
            name.lower() == t_name.lower() or name.lower() == "merge"
        ):
            for f in filenames:
                if "covered" in f:
                    continue

                split = f.rsplit("-")[-1].rsplit(".")[0]
                files[split].append((os.path.join(dirpath, f), t_name, split))

    files = dict(files)
    files = [item for sublist in list(files.values()) for item in sublist]

    return files

@dataclass
class Tree:
    """A class for holding a single tree.
    """

    raw: List = field(default_factory=lambda: [])
    tokens: List = field(default_factory=lambda: [])
    lemmas: List = field(default_factory=lambda: [])
    morph_tags: List = field(default_factory=lambda: [])

    def add(self, branch: List):
        _, word_form, lemma, _, _, morph_tags, _, _, _, _ = branch

        self.raw.append((word_form, lemma, morph_tags))
        self.tokens.append(word_form)
        self.lemmas.append(lemma)
        self.morph_tags.append(morph_tags.rsplit(";"))

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i: int):
        return self.raw[i]

    def __str__(self):
        return f"Tree({self.raw})"

    def __repr__(self):
        return self.__str__()


@dataclass
class Document:
    """A class for holding a single text document.
    """

    sent_id: Union[str, None] = None
    split: Union[str, None] = None
    treebank: Union[str, None] = None
    text: Union[str, None] = None
    tree: List = field(default_factory=lambda: Tree())

    def __str__(self):
        return f"Doc(sent_id={self.sent_id})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.tree.tokens)

    @property
    def tokens(self):
        return self.tree.tokens

    @property
    def lemmas(self):
        return self.tree.lemmas

    @property
    def morph_tags(self):
        return self.tree.morph_tags

    def set_tensors(
        self, chars_tensor, tokens_tensor, morph_tags_tensor, morph_cats_tensor
    ):

        self.chars_tensor = chars_tensor

        self.tokens_tensor = tokens_tensor

        self.morph_tags_tensor = morph_tags_tensor

        self.morph_cats_tensor = morph_cats_tensor

    def set_morph_cats(self, cats_list: List):

        self.morph_cats = cats_list

    def set_lemma_tags(self, tags_tensor: torch.LongTensor):

        self.lemma_tags_tensor = tags_tensor

    # ! Deprecated method
    # TODO (ivo): remove
    def set_word_embeddings(
        self, word_emb: torch.Tensor, word_vec_type: str = "Undefined"
    ):

        self.word_emb_type = word_vec_type
        self.word_emb = word_emb

    # ! Deprecated method
    # TODO (ivo): remove
    def set_context_embeddings(
        self, context_emb: torch.Tensor, model_name: str = "Undefined"
    ):

        self.model_name = model_name
        self.context_emb = context_emb

    def set_pretrained_embeddings(self, x: torch.Tensor) -> None:

        self._pretrained_embeddings = x

    @property
    def pretrained_embeddings(self):

        if hasattr(self, "_pretrained_embeddings"):
            return self._pretrained_embeddings
        else:
            return 0


@dataclass
class DocumentCorpus(Dataset):
    """A class for reading, holding and processing many documents.
    """

    docs: List = field(default_factory=lambda: [])
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    treebanks: Dict = field(default_factory=lambda: defaultdict(int))
    splits: Dict = field(default_factory=lambda: defaultdict(list))
    batch_first: bool = False
    sorted: bool = True
    return_tokens_raw: bool = False

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i: int):
        return self.docs[i]

    def __str__(self):
        return f"DocumentCorpus(\n\tlen={len(self.docs)},\n\ttreebanks={self.treebanks.keys()},\n\tsplits={self.splits.keys()},\n\tbatch_first={self.batch_first},\n\tsorted={self.sorted}\n)"

    def __repr__(self):
        return self.__str__()

    def clear_docs(self):

        self.docs = []

    def _get_vocabs(self):

        self.token_vocab = build_vocab_from_iterator(
            [[t for t in d.tokens] for d in self.docs],
            specials=[self.unk_token, self.pad_token],
            special_first=True,
        )
        self.token_vocab.set_default_index(self.token_vocab[self.unk_token])

        self.char_vocab = build_vocab_from_iterator(
            [[c for c in t] for d in self.docs for t in d.tokens],
            specials=[self.unk_token, self.pad_token],
            special_first=True,
        )
        self.char_vocab.set_default_index(self.token_vocab[self.unk_token])

        self.morph_tag_vocab = {
            k: v
            for v, k in enumerate(
                sorted(
                    {
                        tag
                        for d in self.docs
                        for tagset in d.morph_tags
                        for tag in tagset
                    }
                )
            )
        }

        # ======================================================================
        # Morphological feature categories (for regularization)
        # ======================================================================
        with open("./morphological_tagging/data/uni_morph_tags.json", "rb") as f:
            uni_morph_tags = json.load(f)

        valid_mtags = (
            set(map(lambda x: x.lower(), self.morph_tag_vocab.keys()))
            & uni_morph_tags.keys()
        )

        self.morph_tag_name_vocab = {
            mtag: uni_morph_tags[mtag][0] for mtag in list(valid_mtags)
        }
        self.morph_tag_cat_vocab = {
            mtag: uni_morph_tags[mtag][1] for mtag in list(valid_mtags)
        }
        self.morph_tag_cat_vocab["_"] = "_"

        self.morph_cat_vocab = {
            k: v for v, k in enumerate(sorted(set(self.morph_tag_cat_vocab.values())))
        }
        self.morph_cat_vocab["_"] = len(self.morph_cat_vocab) - 1

    def _move_to_pt(self):

        for d in self.docs:
            chars_tensor = [
                torch.tensor(
                    self.char_vocab.lookup_indices([c for c in t]), dtype=torch.long
                )
                for t in d.tokens
            ]

            tokens_tensor = torch.tensor(
                self.token_vocab.lookup_indices([t for t in d.tokens]), dtype=torch.long
            )

            morph_tags_tensor = torch.stack(
                [
                    torch.sum(
                        F.one_hot(
                            torch.tensor(
                                [self.morph_tag_vocab.get(tag, "_") for tag in tagset],
                                dtype=torch.long,
                            ),
                            len(self.morph_tag_vocab),
                        )[:, :-1],
                        dim=0,
                    )
                    for tagset in d.morph_tags
                ],
                dim=0,
            )

            morph_cats_tensor = torch.stack(
                [
                    torch.sum(
                        F.one_hot(
                            torch.tensor(
                                list(
                                    {
                                        self.morph_cat_vocab[
                                            self.morph_tag_cat_vocab[tag.lower()]
                                        ]
                                        for tag in tagset
                                    }
                                ),
                                dtype=torch.long,
                            ),
                            len(self.morph_cat_vocab),
                        ),
                        dim=0,
                    )
                    for tagset in d.morph_tags
                ],
                dim=0,
            )

            d.set_tensors(
                chars_tensor, tokens_tensor, morph_tags_tensor, morph_cats_tensor
            )

    def parse_tree_file(self, fp: str, treebank_name: str = None, split: str = None):
        """Parse a single document with CONLL-U trees into a list of Documents.
        Will append to documents, not overwrite.

        Args:
            fp (str): filepath to the document.
            treebank_name (str): the name of the treebank which is being parsed

        """
        with open(fp, newline="\n", encoding="utf8") as csvfile:
            conllu_data = csv.reader(csvfile, delimiter="\t", quotechar="\u2400")

            cur_doc = Document()
            for i, row in enumerate(conllu_data):

                # New sentence
                if len(row) == 0:

                    self.docs.append(cur_doc)
                    self.treebanks[
                        treebank_name if treebank_name is not None else "Undefined"
                    ] += 1

                    cur_doc = Document()
                    cur_doc.split = split
                    cur_doc.treebank = treebank_name
                    self.splits[split if split is not None else "Undefined"].append(
                        len(self.docs) - 1
                    )

                # Get sentence ID
                elif "# sent_id = " in row[0]:
                    sent_id = row[0][12:]

                    cur_doc.sent_id = sent_id

                # Get sentence in plain language (non-tokenized)
                elif "# text = " in row[0]:
                    full_text = row[0][9:]

                    cur_doc.text = full_text

                # Get tree information
                # CONLL-U format
                elif row[0].isnumeric():
                    if "." in row[0]:
                        continue
                    cur_doc.tree.add(row)

            if cur_doc.sent_id is not None:
                self.docs.append(cur_doc)

                self.treebanks[
                    treebank_name if treebank_name is not None else "Undefined"
                ] += 1

    def setup(self):
        """Code to run when finished importing all files.
        """
        # Iterate over the documents to get the necessary vocabs
        self._get_vocabs()

        # Iterate over documents again to add the morph. cats. to each doc
        for d in self.docs:
            morph_cats = [
                {self.morph_tag_cat_vocab[tag.lower()] for tag in tagset}
                for tagset in d.morph_tags
            ]

            d.set_morph_cats(morph_cats)

        # Move documents information to tensors for minibatching
        self._move_to_pt()

        # Collect information about documents
        self.treebanks = dict(self.treebanks)
        self.splits = dict(self.splits)

        if self.sorted:
            self.splits["train"] = sorted(
                self.splits["train"], key=lambda x: len(self.docs[x]), reverse=True
            )

            self.splits["dev"] = sorted(
                self.splits["dev"], key=lambda x: len(self.docs[x]), reverse=True
            )

            self.splits["test"] = sorted(
                self.splits["test"], key=lambda x: len(self.docs[x]), reverse=True
            )

    def set_lemma_tags(self):

        # Iterate over all documents once to get stats on the lemma scripts
        self.script_counter, self.script_examples = Counter(), defaultdict(set)
        docs_scripts = []
        for i, d in enumerate(self.docs):

            doc_scripts = []
            for wf, lm in zip(d.tokens, d.lemmas):

                lemma_script = LemmaScriptGenerator(wf, lm).get_lemma_script()
                self.script_counter[lemma_script] += 1

                doc_scripts.append(lemma_script)

                if len(self.script_examples[lemma_script]) < 3:
                    self.script_examples[lemma_script].add(f"{wf}\u2192{lm}")

            docs_scripts.append(doc_scripts)

        # Generate script to class conversion
        self.script_to_id = {
            k: i
            for i, (k, _) in enumerate(
                sorted(self.script_counter.items(), key=lambda x: x[1], reverse=True)
            )
        }

        self.id_to_script = list(self.script_to_id.keys())

        # Add the scripts as classes to the individual documents
        for i, doc_scripts in enumerate(docs_scripts):

            self.docs[i].set_lemma_tags(
                torch.tensor(
                    [self.script_to_id[script] for script in doc_scripts],
                    dtype=torch.long,  # TODO (ivo) add device support
                )
            )

    def lemma_tags_overview(self, n: int = 11) -> pd.DataFrame:
        """Get the most common lemma scripts and some examples.

        Args:
            n (int, optional): number of scripts to show. Defaults to 11.
        """

        most_common_rules = [
            [script, count] for script, count in self.script_counter.most_common(n)
        ]

        for entry in most_common_rules:
            entry.append(self.script_examples[entry[0]])

        df = pd.DataFrame(most_common_rules, columns=["Rule", "Count", "Examples"])

        return df

    def collate_batch(self, batch) -> Tuple[torch.Tensor]:

        docs_subset = [
            [
                d.chars_tensor,
                d.tokens,
                d.tokens_tensor,
                d.pretrained_embeddings,
                d.morph_tags_tensor,
                d.lemma_tags_tensor,
                d.morph_cats_tensor,
            ]
            for d in batch
        ]

        (
            chars,
            tokens_raw,
            tokens,
            pretrained_embeddings,
            morph_tags,
            lemma_tags,
            morph_cats,
        ) = list(map(list, zip(*docs_subset)))

        # Characters [T_c, B]
        char_lens = [c.size(0) for seq in chars for c in seq]

        chars = pad_sequence(
            [c for seq in chars for c in seq],
            padding_value=self.char_vocab[self.pad_token],
        )

        # Tokens [T_t, B]
        token_lens = [seq.size(0) for seq in tokens]

        tokens_raw = [[token for token in seq] for seq in tokens_raw]

        tokens = pad_sequence(tokens, padding_value=0)

        if self.pretrained_embeddings_dim != 0:
            pretrained_embeddings = pad_sequence(pretrained_embeddings, padding_value=0)
        else:
            pretrained_embeddings = None

        # Tags [T_t x B, C]/[T_t x B]
        lemma_tags = pad_sequence(
            lemma_tags, batch_first=self.batch_first, padding_value=-1
        )

        morph_tags = pad_sequence(
            morph_tags, batch_first=self.batch_first, padding_value=-1
        )

        morph_cats = pad_sequence(
            morph_cats, batch_first=self.batch_first, padding_value=-1
        )

        if self.return_tokens_raw:
            return (
                char_lens,
                chars,
                token_lens,
                tokens_raw,
                tokens,
                pretrained_embeddings,
                lemma_tags,
                morph_tags,
                morph_cats,
            )

        else:
            return (
                char_lens,
                chars,
                token_lens,
                tokens,
                pretrained_embeddings,
                lemma_tags,
                morph_tags,
                morph_cats,
            )

    def _collate_batch_preprocess(self, batch) -> Tuple[torch.Tensor]:

        docs_subset = [[d.tokens, d] for d in batch]

        tokens_raw, docs = list(map(list, zip(*docs_subset)))

        token_lens = [len(seq) for seq in tokens_raw]

        tokens_raw = [[token for token in seq] for seq in tokens_raw]

        return token_lens, tokens_raw, docs

    @property
    def pretrained_embeddings_dim(self) -> int:

        if len(self.docs) == 0:
            raise ValueError("This corpus has no documents.")

        else:
            try:
                if isinstance(self.docs[0].pretrained_embeddings, torch.Tensor):
                    return self.docs[0].pretrained_embeddings.size(-1)
                elif self.docs[0].pretrained_embeddings == 0:
                    return 0
            except ValueError:
                return 0

    # ! Deprecated method
    # TODO (ivo): remove
    def add_word_embs(
        self, vecs: Vectors, lower_case_backup: bool = False, **kwargs
    ):
        """Add pre-trained word embeddings to a collection of documents.

        Args:
            docs (List[Document]): [description]
            vecs (Vectors, optional): [description]. Defaults to FastText.
            lower_case_backup (bool, optional): [description]. Defaults to False.

        """

        embeds = vecs(**kwargs)

        for d in self.docs:
            d.set_word_embeddings(
                embeds.get_vecs_by_tokens(d.tokens, lower_case_backup), vecs.__name__
            )

    # ! Deprecated method
    # TODO (ivo): remove
    def add_context_embs(self, model, tokenizer):
        """Generate contextual embedding from document text.

        Args:
            d (Document): [description]
            tokenizer (Huggingface Tokenizer):
            model (Huggingface Transformer):

        """

        for d in self.docs:
            # Tokenize the whole text
            s_tokenized = tokenizer(d.text, return_offsets_mapping=True)

            # Find the token spans within the text
            end, spans = 0, []
            for t in d.tokens:
                match = re.search(re.escape(t), d.text[end:])

                spans.append((match.span()[0] + end, match.span()[1] + end))

                end += match.span()[-1]

            # Find correspondence of tokenized string and dataset tokens
            index, correspondence = 0, defaultdict(list)
            for i, (_, tokenized_end) in enumerate(s_tokenized["offset_mapping"][1:-1]):
                # Iterate through the offset_mapping of the tokenizer
                # add the tokenized token to the mapping for the original token
                correspondence[index].append(i)

                # Increment the index if tokenized token span is exhausted
                if tokenized_end == spans[index][-1]:
                    index += 1

            # Convert from defaultdict to regular dict
            correspondence = dict(correspondence)

            # Get contextualized embeddings
            contextual_embeddings = model(
                **tokenizer(d.text, return_tensors="pt"), output_hidden_states=True
            )

            contextual_embeddings = torch.stack(
                contextual_embeddings["hidden_states"][-4:]
            )
            contextual_embeddings = torch.mean(contextual_embeddings, dim=0).squeeze()

            contextual_embeddings_corresponded = torch.stack(
                [
                    torch.mean(contextual_embeddings[correspondence[k]], dim=0)
                    for k in correspondence.keys()
                ]
            )

            d.set_context_embeddings(
                contextual_embeddings_corresponded, type(model).__name__
            )


class TreebankDataModule(pl.LightningDataModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """

    def __init__(
        self,
        batch_size,
        language: Optional[str] = None,
        treebank_name: Optional[str] = None,
        batch_first: Optional[bool] = True,
        len_sorted: Optional[bool] = True,
        unk_token: Optional[str] = "<UNK>",
        pad_token: Optional[str] = "<PAD>",
        return_tokens_raw: Optional[bool] = True,
    ):
        super().__init__()

        # TODO (ivo): add support for predefined data module loading
        # if (language is None) or (sorted is None) and :
        #    raise ConfigurationError

        self.language = language
        self.treebank_name = treebank_name
        self.batch_first = batch_first
        self.len_sorted = len_sorted
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.return_tokens_raw = return_tokens_raw
        self.batch_size = batch_size

    def prepare_data(self) -> None:

        files = get_conllu_files(language=self.language, name=self.treebank_name,)

        self.corpus = DocumentCorpus(
            batch_first=self.batch_first,
            sorted=self.len_sorted,
            return_tokens_raw=self.return_tokens_raw,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
        )

        for (fp, t_name, split) in files:
            self.corpus.parse_tree_file(fp, t_name, split)
        self.corpus.setup()

        self.corpus.set_lemma_tags()

        print(self.corpus)

    def setup(self, stage: Optional[str] = None):

        if stage in (None, "fit"):
            self.train_corpus = Subset(self.corpus, self.corpus.splits["train"])

            self.valid_corpus = Subset(self.corpus, self.corpus.splits["dev"])

        if stage in (None, "test"):
            self.test_corpus = Subset(self.corpus, self.corpus.splits["test"])

    def _preprocess_dataloader(self, batch_size: Optional[int] = None):

        if batch_size is None:
            batch_size = self.batch_size

        data_loader = DataLoader(
            self.corpus,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.corpus._collate_batch_preprocess,
        )

        return data_loader

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_corpus,
            batch_size=self.batch_size,
            shuffle=False if self.len_sorted else True,
            collate_fn=self.corpus.collate_batch,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.valid_corpus,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.corpus.collate_batch,
        )

        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_corpus,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.corpus.collate_batch,
        )

        return test_loader
