from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import pytorch_lightning as pl

from morphological_tagging.metrics import clf_metrics
from morphological_tagging.models.modules import (
    char2word,
    residual_lstm,
    residual_mlp_layer,
)
from utils.common_operations import label_smooth


class UDPipe2(pl.LightningModule):
    """A PyTorch Lightning implementation of UDPipe2.0.

    As described in:
        Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: \n
        Contextualized embeddings, regularization with morphological categories, corpora merging. \n
        arXiv preprint arXiv:1908.06931.

    """

    def __init__(
        self,
        char_vocab,
        token_vocab,
        c2w_kwargs: dict,
        n_lemma_scripts: int,
        n_morph_tags: int,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        w_embedding_dim: int = 512,
        pretrained_embedding_dim: int = 300,
        udpipe_rnn_dim: int = 512,
        udpipe_bidirectional: bool = True,
        dropout_p: float = 0.5,
        token_mask_p: float = 0.2,
        label_smoothing: float = 0.03,
        lr: float = 1e-3,
        betas: Tuple[float] = (0.9, 0.99),
        scheduler_name: Tuple[str, None] = None,
        scheduler_kwargs: Tuple[dict, None] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.c2w_kwargs = c2w_kwargs
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.w_embedding_dim = w_embedding_dim
        self.pretrained_embedding_dim = pretrained_embedding_dim
        self.udpipe_rnn_dim = udpipe_rnn_dim
        self.udpipe_bidirectional = udpipe_bidirectional
        self.dropout_p = dropout_p
        self.token_mask_p = token_mask_p
        self.label_smoothing = label_smoothing
        self.lr = lr
        self.betas = betas
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs  = scheduler_kwargs

        # ======================================================================
        # Embeddings
        # ======================================================================
        self.c2w_embeddings = char2word(
            vocab_len=len(char_vocab),
            padding_idx=char_vocab[self.pad_token],
            **self.c2w_kwargs,
        )

        c2w_out_dim = self.c2w_kwargs["out_dim"]

        self.w_embeddings = nn.Embedding(
            len(token_vocab),
            embedding_dim=self.w_embedding_dim,
            padding_idx=token_vocab[self.pad_token],
        )

        self.embed_dropout = nn.Dropout(p=self.dropout_p)

        # ======================================================================
        # Word-level recurrent layers
        # ======================================================================
        self.udpipe_input_dim = (
            c2w_out_dim + self.w_embedding_dim + self.pretrained_embedding_dim
        )
        self.udpipe_output_dim = (
            2 if self.udpipe_bidirectional else 1
        ) * self.udpipe_rnn_dim

        self.rnn_1 = residual_lstm(
            input_size=self.udpipe_input_dim,
            hidden_size=self.udpipe_rnn_dim,
            bidirectional=self.udpipe_bidirectional,
            dropout_p=self.dropout_p,
            residual=False,
        )

        # Dropout for residual lstms are handled inside class
        # Avoids dropping out the residual connection
        self.rnn_2 = residual_lstm(
            input_size=self.udpipe_output_dim,
            hidden_size=self.udpipe_rnn_dim,
            bidirectional=self.udpipe_bidirectional,
            dropout_p=self.dropout_p,
        )

        self.rnn_3 = residual_lstm(
            input_size=self.udpipe_output_dim,
            hidden_size=self.udpipe_rnn_dim,
            bidirectional=self.udpipe_bidirectional,
            dropout_p=self.dropout_p,
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
                out_features=self.n_lemma_scripts,
            ),
        )

        self.morph_feature_extractor = residual_mlp_layer(
            in_features=self.udpipe_output_dim, out_features=self.udpipe_output_dim
        )

        self.morph_joint_classifier = nn.Linear(
            in_features=self.udpipe_output_dim, out_features=self.n_morph_tags
        )

        self.morph_single_classifiers = nn.ModuleList(
            [
                nn.Linear(in_features=self.udpipe_output_dim, out_features=1)
                for i in range(self.n_morph_tags)
            ]
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.unk_token_idx = token_vocab[unk_token]
        self.token_mask = D.bernoulli.Bernoulli(1 - torch.tensor([token_mask_p]))

        self.configure_metrics()

    def configure_metrics(self):

        self.morph_metrics_train = clf_metrics(
            K=self.n_morph_tags, prefix="morph_train"
        )
        self.morph_metrics_valid = clf_metrics(
            K=self.n_morph_tags, prefix="morph_valid"
        )
        self.morph_metrics_test = clf_metrics(K=self.n_morph_tags, prefix="morph_test")

        self.lemma_metrics_train = clf_metrics(
            K=self.n_lemma_scripts, prefix="lemma_train"
        )
        self.lemma_metrics_valid = clf_metrics(
            K=self.n_lemma_scripts, prefix="lemma_valid"
        )
        self.lemma_metrics_test = clf_metrics(
            K=self.n_lemma_scripts, prefix="lemma_test"
        )

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

        if self.scheduler_name is None:
            return [optimizer]

        elif self.scheduler_name.lower() == "step":
            scheduler = MultiStepLR(optimizer, **self.scheduler_kwargs)

        elif self.scheduler_name.lower() == "plateau":
            scheduler_ = ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)

            scheduler = {
                "scheduler": scheduler_,
                "reduce_on_plateau": True,
                "monitor": "Valid Accuracy",
                "interval": "epoch",
                "name": "LR Reduce on Plateau",
            }

        return [optimizer], [scheduler]

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        if isinstance(char_lens, list):
            char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

        if isinstance(token_lens, list):
            token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

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
        # Replace token with <UNK> with a certain probability
        tokens_ = torch.where(
            self.token_mask.sample(tokens.size()).squeeze().bool().to(self.device),
            tokens,
            self.unk_token_idx,
        )

        w_embeds = self.w_embeddings(tokens_)

        # Concatenate all embeddings together
        embeds = torch.cat([c2w_embs, w_embeds, pretrained_embeddings], dim=-1)

        embeds = self.embed_dropout(embeds)

        # ======================================================================
        # Word-level recurrent layers
        # ======================================================================
        # Pass the word embeddings through the LSTMs
        embeds = pack_padded_sequence(embeds, token_lens, enforce_sorted=False)

        # Input to hidden, no residual connection
        h_t = self.rnn_1(embeds)

        # Hidden to hidden, residual connection
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

        lemma_loss = F.cross_entropy(
            lemma_script_logits, lemma_tags, label_smoothing=self.label_smoothing
        )
        morph_loss = F.binary_cross_entropy_with_logits(
            morph_logits, label_smooth(self.label_smoothing, morph_tags)
        )

        loss = lemma_loss + morph_loss
        losses = {"lemma": lemma_loss, "morph": morph_loss}

        if morph_reg_logits is not None:
            morph_reg_loss = sum(
                F.binary_cross_entropy_with_logits(
                    logits,
                    label_smooth(self.label_smoothing, morph_tags[:, i].unsqueeze(-1)),
                )
                for i, logits in enumerate(morph_reg_logits)
            )

            loss += morph_reg_loss
            losses["morph_reg"] = morph_reg_loss

        losses["total"] = loss

        return loss, losses

    def training_step(self, batch, batch_idx):
        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            morph_tags,
            lemma_tags,
        ) = batch[0]

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits, morph_logits, lemma_tags, morph_tags, morph_reg_logits
        )

        self.log_dict({f"{k}_loss_train": v for k, v in losses.items()})
        self.log_dict(
            self.lemma_metrics_train(torch.softmax(lemma_logits, dim=-1), lemma_tags)
        )
        self.log_dict(
            self.morph_metrics_train(torch.softmax(morph_logits, dim=-1), morph_tags)
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            morph_tags,
            lemma_tags,
        ) = batch

        lemma_logits, morph_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(lemma_logits, morph_logits, lemma_tags, morph_tags)

        self.log_dict({f"{k}_loss_valid": v for k, v in losses.items()})
        self.log_dict(
            self.lemma_metrics_valid(torch.softmax(lemma_logits, dim=-1), lemma_tags)
        )
        self.log_dict(
            self.morph_metrics_valid(torch.softmax(morph_logits, dim=-1), morph_tags)
        )

        return loss

    def test_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            morph_tags,
            lemma_tags,
        ) = batch

        lemma_logits, morph_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(lemma_logits, morph_logits, lemma_tags, morph_tags)

        self.log_dict({f"{k}_loss_test": v for k, v in losses.items()})
        self.log_dict(
            self.lemma_metrics_test(torch.softmax(lemma_logits, dim=-1), lemma_tags)
        )
        self.log_dict(
            self.morph_metrics_test(torch.softmax(morph_logits, dim=-1), morph_tags)
        )

        return loss
