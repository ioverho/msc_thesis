import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors

from transformers import AutoConfig, AutoTokenizer, AutoModel
from utils.errors import ConfigurationError

FASTTEXT_LANG_CONVERSION = {
    "Arabic": "ar",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "Turkish": "tr",
}

class FastText(Vectors):
    """Slightly rewritten FastText vector class to use multi-lingual embeddings.
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


class UDPipe2PreProcessor(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(
        self,
        word_embeddings: bool = True,
        context_embeddings: bool = True,
        language: Optional[str] = None,
        cache_location: str = "./morphological_tagging/data/pretrained_vectors",
        lower_case_backup: bool = False,
        transformer_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        ) -> None:
        super().__init__()

        self.word_embeddings = word_embeddings
        self.context_embeddings = context_embeddings
        self.language = language
        self.cache_location = cache_location
        self.lower_case_backup = lower_case_backup
        self.transformer_name = transformer_name
        self.batch_size = batch_size

        if self.word_embeddings:
            self.vecs = FastText(
                language=FASTTEXT_LANG_CONVERSION[language],
                cache=cache_location
                )

        if self.context_embeddings:
            config = AutoConfig.from_pretrained(self.transformer_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.transformer_name,
                use_fast=True)
            self.transformer = AutoModel.from_config(config)

            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _word_embeddings(self, token_lens, tokens_raw):
        word_embeddings_ = self.vecs.get_vecs_by_tokens(
            [t for seq in tokens_raw for t in seq],
            self.lower_case_backup
            )

        beg = 0
        word_embeddings = []
        for l in token_lens:
            word_embeddings.append(word_embeddings_[beg:beg+l, :])
            beg += l

        return word_embeddings

    @torch.no_grad()
    def _context_embeddings(self, token_lens, tokens_raw):

        transformer_input = self.tokenizer(
            tokens_raw,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        token_map = [
            torch.logical_and(
                transformer_input["offset_mapping"][i, :, 0]
                == 0,  # Only keep the first BPE, i.e. those with non-zero span start
                transformer_input["offset_mapping"][i, :, 1]
                != 0,  # Remove [CLS], [END], [PAD] tokens, i.e. those with non-zero span end
            )
            for i in range(len(tokens_raw))
        ]

        transformer_output = self.transformer(
            transformer_input["input_ids"],
            transformer_input["attention_mask"],
            output_hidden_states=False,
        ).last_hidden_state

        context_embeddings = [
            transformer_output[i, token_map[i], :]
            for i in range(len(tokens_raw))
            ]

        return context_embeddings

    def forward(self, batch: Union[tuple, List[List[str]], str], set_doc_attr: bool = False):

        if isinstance(batch, tuple):
            (
                token_lens,
                tokens_raw,
                docs
            ) = batch

        elif isinstance(batch, list):
            tokens_raw = batch
            token_lens = [len(seq) for seq in tokens_raw]
            set_doc_attr = False

        elif isinstance(batch, str):
            tokens_raw = [[batch]]
            token_lens = [len(batch)]
            set_doc_attr = False

        embeddings_ = []

        if self.word_embeddings:
            word_embeddings = self._word_embeddings(token_lens, tokens_raw)
            embeddings_.append(word_embeddings)

        if self.context_embeddings:
            context_embeddings = self._context_embeddings(token_lens, tokens_raw)
            embeddings_.append(context_embeddings)

        if len(embeddings_) == 2:
            embeddings = [
                torch.cat([w_seq, c_seq.to("cpu")], dim=-1)
                for w_seq, c_seq in zip(*embeddings_)
                ]

        elif len(embeddings_) == 1:
            embeddings = embeddings_[0]

        if set_doc_attr:
            for d, e in zip(docs, embeddings):
                d.set_pretrained_embeddings(e.detach().cpu())

        return embeddings

    @property
    def device(self):

        return next(self.model_shared.parameters()).device
