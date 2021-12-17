import os
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator, Vectors
import pandas as pd

from morphological_tagging.data.lemma_script import LemmaScriptGenerator


BASEPATH = "./morphological_tagging/data/sigmorphon_2019/task2"


FASTTEXT_LANG_CONVERSION = {
    "Arabic": "ar",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "Turkish": "tr"
}


def get_conllu_files(language: str, name: str = 'merge', splits: bool = True) -> Union[dict[list], list]:
    """Get the UD/CONLLU/SIGMORPHON treebanks from the BASEPATH dir.

    Args:
        language (str): the desired language
        name (str): the desired treebank name, or merge for all. Default to "merge"

    Returns:
        Union[dict[list], list]: a dict, with as keys the data split and as values a list with filepaths.
            If `splits` is False, then return a list.
    """

    files = defaultdict(list)
    for (dirpath, dirnames, filenames) in os.walk(BASEPATH):
        if len(filenames) == 0:
            continue

        treebank = os.path.split(dirpath)[-1]
        _, treebank = re.split(r"_", treebank, maxsplit=1)
        t_lang, t_name = re.split(r"-", treebank, maxsplit=1)

        if t_lang.lower() == language.lower() and (name.lower() == t_name.lower() or name.lower() == "merge"):
            for f in filenames:
                if 'covered' in f:
                    continue

                split = f.rsplit("-")[-1].rsplit(".")[0]
                files[split].append((
                    os.path.join(dirpath, f),
                    t_name,
                    split
                    ))

    files = dict(files)

    if not splits:
        files = [item for sublist in list(files.values()) for item in sublist]

    return files


class FastText(Vectors):
    """Slightly rewritten FastText vecot class to use multi-lingual embeddings.
    """

    url_base = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"

    def __init__(self, language="en", **kwargs):

        if len(language) > 2:
            language = FASTTEXT_LANG_CONVERSION.get(language)
            if language is None:
                raise ValueError(f"Language acronym '{language}' not recognized.")

        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


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

    def set_tensors(self, chars_tensor, tokens_tensor, morph_tags_tensor):

        self.chars_tensor = chars_tensor

        self.tokens_tensor = tokens_tensor

        self.morph_tags_tensor = morph_tags_tensor

    def set_lemma_tags(self, tags_tensor: torch.LongTensor):

        self.lemma_tags_tensor = tags_tensor

    def set_word_embeddings(
        self, word_emb: torch.Tensor, word_vec_type: str = "Undefined"
    ):

        self.word_emb_type = word_vec_type
        self.word_emb = word_emb

    def set_context_embeddings(
        self, context_emb: torch.Tensor, model_name: str = "Undefined"
    ):

        self.model_name = model_name
        self.context_emb = context_emb

    @property
    def pretrained_embeddings(self):
        pretrained_embeds = []

        if hasattr(self, "word_emb"):
            pretrained_embeds.append(self.word_emb)

        if hasattr(self, "context_emb"):
            pretrained_embeds.append(self.context_emb)

        if len(pretrained_embeds) == 0:
            raise ValueError("Document has no pretrained embeddings.")
        else:
            pretrained_embeds = torch.cat(pretrained_embeds, dim=-1)
            # TODO (ivo): add device support

        return pretrained_embeds


@dataclass
class DocumentCorpus(Dataset):
    """A class for reading, holding and processing many documents.
    """

    docs: List = field(default_factory=lambda: [])
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    treebanks: Dict = field(default_factory=lambda: defaultdict(int))
    splits: Dict = field(default_factory=lambda: defaultdict(list))

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i: int):
        return self.docs[i]

    def __str__(self):
        return f"DocumentCorpus(len={len(self.docs)})"

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

    def _move_to_pt(self):

        for d in self.docs:
            chars_tensor = [
                torch.tensor(
                    self.char_vocab.lookup_indices([c for c in t]),
                    dtype=torch.long
                    # TODO (ivo): add device support
                )
                for t in d.tokens
            ]

            tokens_tensor = torch.tensor(
                self.token_vocab.lookup_indices([t for t in d.tokens]),
                dtype=torch.long
                # TODO (ivo): add device support
            )

            morph_tags_tensor = torch.stack(
                [
                    torch.sum(
                        F.one_hot(
                            torch.tensor(
                                [self.morph_tag_vocab.get(tag, "_") for tag in tagset],
                                dtype=torch.long
                                # TODO (ivo): add device support
                            ),
                            len(self.morph_tag_vocab),
                        )[:, :-1],
                        dim=0,
                    )
                    for tagset in d.morph_tags
                ],
                dim=0,
            )

            d.set_tensors(chars_tensor, tokens_tensor, morph_tags_tensor)

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
                    self.treebanks[treebank_name if treebank_name is not None else "Undefined"] += 1

                    cur_doc = Document()
                    cur_doc.split = split
                    self.splits[split if split is not None else "Undefined"].append(len(self.docs)-1)

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

                self.treebanks[treebank_name if treebank_name is not None else "Undefined"] += 1

    def setup(self):
        """Code to run when finished importing all files.
        """
        self._get_vocabs()
        self._move_to_pt()

        self.treebanks = dict(self.treebanks)
        self.splits = dict(self.splits)

    def add_word_embs(
        self, vecs: Vectors = FastText, lower_case_backup: bool = False, **kwargs
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
                d.tokens_tensor,
                d.pretrained_embeddings,
                d.morph_tags_tensor,
                d.lemma_tags_tensor,
            ]
            for d in batch
        ]

        chars, tokens, pretrained_embeddings, morph_tags, lemma_tags = list(
            map(list, zip(*docs_subset))
        )

        # Characters [T_c, B]
        char_lens = [c.size(0) for seq in chars for c in seq]

        chars = pad_sequence(
            [c for seq in chars for c in seq],
            padding_value=self.char_vocab[self.pad_token],
        )

        # Tokens [T_t, B]
        token_lens = [seq.size(0) for seq in tokens]

        tokens = pad_sequence(tokens, padding_value=0)

        pretrained_embeddings = pad_sequence(pretrained_embeddings, padding_value=0)

        # Tags [T_t x B, C]/[T_t x B]
        morph_tags = torch.cat(morph_tags)

        lemma_tags = torch.cat(lemma_tags)

        return char_lens, chars, token_lens, tokens, pretrained_embeddings, morph_tags, lemma_tags

    @property
    def pretrained_embeddings_dim(self):

        if len(self.docs) == 0:
            raise ValueError("This corpus has no documents.")

        else:
            try:
                return self.docs[0].pretrained_embeddings.size(-1)
            except ValueError:
                return 0