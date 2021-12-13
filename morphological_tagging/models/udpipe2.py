from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pytorch_lightning as pl

from morphological_tagging.models.modules import (
    char2word,
    residual_lstm,
    residual_mlp_layer,
)


class UDPipe2(pl.LightningModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """

    def __init__(
        self,
        char_vocab,
        token_vocab,
        c2w_kwargs: dict,
        n_lemma_scripts: int,
        n_morph_tags: int,
        pad_token: int = 1,
        w_embedding_dim: int = 512,
        pretrained_embedding_dim: int = 300,
        udpipe_bidirectional: bool = True,
        udpipe_rnn_dim: int = 512,
    ) -> None:
        super().__init__()

        # ======================================================================
        # Embeddings
        # ======================================================================
        self.c2w_embeddings = char2word(vocab_len=len(char_vocab), **c2w_kwargs)

        c2w_out_dim = c2w_kwargs["out_dim"]

        self.w_embeddings = nn.Embedding(
            len(token_vocab),
            embedding_dim=w_embedding_dim,
            padding_idx=token_vocab[pad_token],
        )

        # ======================================================================
        # Word-level recurrent layers
        # ======================================================================
        self.udpipe_input_dim = (
            c2w_out_dim + w_embedding_dim + pretrained_embedding_dim
        )
        self.udpipe_output_dim = (2 if udpipe_bidirectional else 1) * udpipe_rnn_dim

        self.rnn_1 = nn.LSTM(
            input_size=self.udpipe_input_dim,
            hidden_size=udpipe_rnn_dim,
            bidirectional=udpipe_bidirectional,
        )

        self.rnn_2 = residual_lstm(
            input_size=self.udpipe_output_dim,
            hidden_size=udpipe_rnn_dim,
            bidirectional=udpipe_bidirectional,
        )

        self.rnn_3 = residual_lstm(
            input_size=self.udpipe_output_dim,
            hidden_size=udpipe_rnn_dim,
            bidirectional=udpipe_bidirectional,
        )

        # ======================================================================
        # Classifiers
        # ======================================================================
        self.lemma_script_classifier = nn.Sequential(
            residual_mlp_layer(
                in_features=self.udpipe_output_dim + c2w_out_dim,
                out_features=self.udpipe_output_dim + c2w_out_dim,
            ),
            nn.Linear(
                in_features=self.udpipe_output_dim + c2w_out_dim,
                out_features=n_lemma_scripts,
            ),
        )

        self.morph_feature_extractor = residual_mlp_layer(
            in_features=self.udpipe_output_dim, out_features=self.udpipe_output_dim
        )

        self.morph_joint_classifier = nn.Linear(
            in_features=self.udpipe_output_dim, out_features=n_morph_tags
        )

        self.morph_single_classifiers = nn.ModuleList(
            [
                nn.Linear(in_features=self.udpipe_output_dim, out_features=1)
                for i in range(n_morph_tags)
            ]
        )

    def forward(
        self,
        char_lens: torch.Tensor,
        chars: torch.Tensor,
        token_lens: torch.Tensor,
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        # ======================================================================
        # Embeddings
        # ======================================================================
        # Get char2word embeddings
        c2w_embs_ = self.c2w_embeddings(chars, char_lens)

        seqs = []
        beg = torch.tensor([0])
        for l in token_lens:
            seqs.append(c2w_embs_[beg : beg + l])
            beg += l

        c2w_embs = pad_sequence(seqs, padding_value=0.0)

        # Get word embeddings
        w_embeds = self.w_embeddings(tokens)

        # Concatenate all embeddings together
        embeds = torch.cat([c2w_embs, w_embeds, pretrained_embeddings], dim=-1)

        # ======================================================================
        # Word-level recurrent layers
        # ======================================================================
        # Pass the word embeddings through the LSTMs
        embeds = pack_padded_sequence(embeds, token_lens, enforce_sorted=False)

        h_t, _ = self.rnn_1(embeds)
        h_t = self.rnn_2(h_t)
        h_t = self.rnn_3(h_t)

        # ======================================================================
        # Classifiers
        # ======================================================================
        # No need for unpacking
        # Using packed 'vector' for token-level sequence classification
        lemma_script_logits = self.lemma_script_classifier(
            torch.cat([c2w_embs_, h_t.data], dim=-1)
        )

        morph_feats = self.morph_feature_extractor(h_t.data)

        morph_logits = self.morph_joint_classifier(morph_feats)

        if self.training:
            morph_reg_logits = [
                clf(morph_feats) for clf in self.morph_single_classifiers
            ]

            return lemma_script_logits, morph_logits, morph_reg_logits

        else:

            return lemma_script_logits, morph_logits

    def loss(
        self,
        lemma_script_logits: torch.Tensor,
        morph_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_reg_logits: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        loss = F.cross_entropy(lemma_script_logits, lemma_tags)
        loss += F.binary_cross_entropy_with_logits(morph_logits, morph_tags.float())

        if morph_reg_logits is not None:
            loss += sum(
                F.binary_cross_entropy_with_logits(
                    logits, morph_tags.float()[:, i].unsqueeze(-1)
                )
                for i, logits in enumerate(morph_reg_logits)
            )

            return loss