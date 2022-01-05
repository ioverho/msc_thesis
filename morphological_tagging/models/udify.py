import random
from typing import Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModel
import pytorch_lightning as pl

from morphological_tagging.metrics import RunningStats, RunningStatsBatch, RunningF1
from morphological_tagging.models.modules import (
    Char2Word,
    ResidualRNN,
    ResidualMLP,
    LayerAttention,
)
from morphological_tagging.models.optim import InvSqrtWithLinearWarmupScheduler
from utils.common_operations import label_smooth


class UDIFY(pl.LightningModule):
    """A PyTorch Lightning implementation of UDPipe2.0.

    As described in:
        Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: \n
        Contextualized embeddings, regularization with morphological categories, corpora merging. \n
        arXiv preprint arXiv:1908.06931.

    """

    def __init__(
        self,
        transformer_type: str,
        transformer_name: str,
        transformer_dropout: float,
        c2w_kwargs: Dict[str, Any],
        layer_attn_kwargs: Dict[str, Any],
        lemma_rnn_kwargs: Dict[str, Any],
        morph_rnn_kwargs: Dict[str, Any],
        label_smoothing: float,
        mask_p: float,
        transformer_lrs: Dict[int, float],
        rnn_lr: float,
        clf_lr: float,
        n_warmup_steps: int,
        optim_kwargs: Dict[str, Any],
        len_char_vocab: int,
        idx_char_pad: int,
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.transformer_type = transformer_type
        self.transformer_name = transformer_name
        self.transformer_dropout = transformer_dropout
        self.c2w_kwargs = c2w_kwargs
        self.layer_attn_kwargs = layer_attn_kwargs
        self.lemma_rnn_kwargs = lemma_rnn_kwargs
        self.morph_rnn_kwargs = morph_rnn_kwargs
        self.label_smoothing = label_smoothing

        # Number of classes ====================================================
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.n_morph_cats = n_morph_cats

        # Transformer & C2W Embeddings =========================================
        self.config = AutoConfig.from_pretrained(
            transformer_name,
            dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
        )
        self.transformer = AutoModel.from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True,)

        self.c2w = Char2Word(
            vocab_len=len_char_vocab, padding_idx=idx_char_pad, **c2w_kwargs,
        )

        # Word-level RNNs ======================================================
        self.lemma_layer_attn = LayerAttention(**layer_attn_kwargs)

        self.lemma_lstm = ResidualRNN(**lemma_rnn_kwargs)

        self.morph_layer_attn = LayerAttention(**layer_attn_kwargs)

        self.morph_lstm = ResidualRNN(**morph_rnn_kwargs)

        # Lemma classification =================================================
        # self._lemma_in_features = (
        #    self.word_rnn_kwargs["h_dim"] + self.c2w_kwargs["out_dim"]
        # )
        self._lemma_in_features = lemma_rnn_kwargs["h_dim"]

        self.lemma_clf = nn.Sequential(
            ResidualMLP(
                in_features=self._lemma_in_features,
                out_features=self._lemma_in_features,
            ),
            nn.Linear(
                in_features=self._lemma_in_features, out_features=self.n_lemma_scripts
            ),
        )

        # Morph classification =================================================
        self._morph_in_features = morph_rnn_kwargs["h_dim"]

        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_tags - 1
            ),
        )

        self.morph_clf_fac = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_cats
            ),
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.mask_p = mask_p

        self.label_smoothing = label_smoothing

        # ======================================================================
        # Optimization
        # ======================================================================
        self.transformer_lrs = transformer_lrs
        self.rnn_lr = rnn_lr
        self.clf_lr = clf_lr
        self.n_warmup_steps = n_warmup_steps
        self.optim_kwargs = optim_kwargs

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        self.configure_metrics()

        # Special tokens =======================================================
        self.idx_char_pad = idx_char_pad
        self.idx_token_pad = idx_token_pad

    def configure_optimizers(self):

        transformer_layers = {
            l: layer
            for l, layer in enumerate(self.transformer._modules["transformer"].layer)
        }
        transformer_layers["embeddings"] = self.transformer._modules["embeddings"]

        lrs = [
            {"params": v.parameters(), "lr": self.transformer_lrs[k]}
            for k, v in transformer_layers.items()
        ]

        lrs.append({"params": self.c2w.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_lstm.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_lstm.parameters(), "lr": self.rnn_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_unf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_fac.parameters(), "lr": self.clf_lr})

        optimizer = AdamW(lrs, **self.optim_kwargs)

        scheduler = InvSqrtWithLinearWarmupScheduler(
            optimizer, default_lrs=lrs, n_warmup_steps=self.n_warmup_steps
        )

        lr_scheduler_config = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return lr_scheduler_config

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens_raw: torch.Tensor,
        skip_morph_reg: bool = False,
    ) -> Tuple[torch.Tensor]:

        # The lens tensors need to be on CPU in case of packing
        if isinstance(char_lens, list):
            char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

        if isinstance(token_lens, list):
            token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

        batch_size = len(tokens_raw)

        # ==============================================================================
        # Encoding
        # ==============================================================================
        if self.mask_p >= 0.0:
            tokens = [
                [
                    t if random.random() >= self.mask_p else self.tokenizer._mask_token
                    for t in seq
                ]
                for seq in tokens_raw
            ]

        bert_input = self.tokenizer(
            tokens_raw,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        token_map = [
            torch.logical_and(
                bert_input["offset_mapping"][i, :, 0]
                == 0,  # Only keep the first BPE, i.e. those with non-zero span start
                bert_input["offset_mapping"][i, :, 1]
                != 0,  # Remove [CLS], [END], [PAD] tokens, i.e. those with non-zero span end
            )
            for i in range(batch_size)
        ]

        bert_output = self.transformer(
            bert_input["input_ids"].to(self.device),
            bert_input["attention_mask"].to(self.device),
            output_hidden_states=True,
            return_dict=True,
        )

        h_bert = torch.stack(bert_output["hidden_states"], dim=2)

        c2w_embeds_ = self.c2w(chars, char_lens)

        seqs = []
        beg = torch.tensor([0])
        for l in token_lens:
            seqs.append(c2w_embeds_[beg : beg + l])
            beg += l

        c2w_embeds = pad_sequence(
            seqs, padding_value=self.idx_token_pad, batch_first=True
        )

        # ==============================================================================
        # Lemma decoder
        # ==============================================================================
        h_lemma = self.lemma_layer_attn(h_bert)

        h_lemma_sliced = pad_sequence(
            [h_lemma[i, token_map[i], :] for i in range(batch_size)], batch_first=True
        )

        h_lemma = self.lemma_lstm(h_lemma_sliced + c2w_embeds)

        lemma_logits = self.lemma_clf(h_lemma)

        # ==============================================================================
        # Morph tag decoder
        # ==============================================================================
        h_morph = self.morph_layer_attn(h_bert)

        h_morph_sliced = pad_sequence(
            [h_morph[i, token_map[i], :] for i in range(batch_size)], batch_first=True
        )

        h_morph = self.morph_lstm(h_morph_sliced + c2w_embeds)

        morph_logits_unf = self.morph_clf_unf(h_morph)

        if not skip_morph_reg:
            morph_logits_fac = self.morph_clf_fac(h_morph)

            return lemma_logits, morph_logits_unf, morph_logits_fac

        return lemma_logits, morph_logits_unf

    def loss(
        self,
        lemma_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_logits_unf: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_logits_fac: Union[torch.Tensor, None] = None,
        morph_cats: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        lemma_loss = F.cross_entropy(
            lemma_logits.permute(0, 2, 1),
            lemma_tags,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != self.ignore_idx])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = F.binary_cross_entropy_with_logits(
                morph_logits_fac,
                label_smooth(self.label_smoothing, morph_cats.float()),
                reduction="none",
            )
            morph_fac_loss = torch.mean(morph_fac_loss[morph_cats != self.ignore_idx])

            loss = lemma_loss + morph_unf_loss + morph_fac_loss
            losses = {
                "total": loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def configure_metrics(self):

        self._metrics_dict = {
            "train": {
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "valid": {
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "test": {
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "predict": {
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
        }

    def clear_metrics(self, split: str):

        self._metrics_dict[split] = {
            "lemma_acc": RunningStatsBatch(),
            "lemma_lev_dist": RunningStats(),
            "morph_tag_acc": RunningStatsBatch(),
            "morph_set_acc": RunningStatsBatch(),
            "morph_f1": RunningF1(),
        }

    @torch.no_grad()
    def metrics(
        self,
        split: str,
        lemma_logits,
        lemma_tags,
        morph_logits,
        morph_tags,
        token_lens=None,
        tokens_raw=None,
    ):

        if (token_lens is not None) and (tokens_raw is not None):
            skip_lev_dist = False
        else:
            skip_lev_dist = True

        ## Lemma CLF metrics
        lemma_preds = torch.argmax(lemma_logits, dim=-1).detach().cpu().numpy()
        lemma_targets = lemma_tags.detach().cpu().numpy()
        lemma_mask = np.where((lemma_tags != -1).detach().cpu().numpy(), 1.0, np.nan)

        if not skip_lev_dist:
            # TODO (ivo): implement lev_distance reporting inside model
            raise NotImplementedError
            # for i, (preds_seq, target_seq, seq_len) in enumerate(
            #    zip(lemma_preds, lemma_targets, token_lens)
            # ):
            #    for pred, target, token in zip(
            #        preds_seq[:seq_len], target_seq[:seq_len], tokens_raw[i]
            #    ):
            #        pred_lemma_script = corpus.id_to_script[pred]
            #        pred_lemma = apply_lemma_script(token, pred_lemma_script)

            #        target_lemma_script = corpus.id_to_script[target]
            #        target_lemma = apply_lemma_script(token, target_lemma_script)

            #        lemma_lev_dist(distance(pred_lemma, target_lemma), output=False)

        self._metrics_dict[split]["lemma_acc"](lemma_preds == lemma_targets, lemma_mask)

        ## Morph. CLF metrics
        morph_preds = torch.round(torch.sigmoid(morph_logits)).detach().cpu().numpy()
        morph_targets = morph_tags.detach().cpu().numpy()
        morph_mask = np.where((morph_tags != -1).detach().cpu().numpy(), 1.0, np.nan)
        morph_set_mask = np.max(morph_mask, axis=-1)

        item_match = morph_preds == morph_targets
        set_match = np.all((morph_preds == morph_targets), axis=-1)

        # Morph. Acc
        self._metrics_dict[split]["morph_tag_acc"](item_match, morph_mask)
        self._metrics_dict[split]["morph_set_acc"](set_match, morph_set_mask)

        self._metrics_dict[split]["morph_f1"](
            morph_preds, morph_targets, morph_set_mask
        )

    def log_metrics(self, split):

        lemma_acc, _, lemma_se = self._metrics_dict[split]["lemma_acc"]._return_stats()
        (lemma_dist, _, lemma_dist_se, _, _,) = self._metrics_dict[split][
            "lemma_lev_dist"
        ]._return_stats()

        morph_tag_acc, _, morph_tag_se = self._metrics_dict[split][
            "morph_tag_acc"
        ]._return_stats()
        morph_set_acc, _, morph_set_se = self._metrics_dict[split][
            "morph_set_acc"
        ]._return_stats()
        (morph_precision, morph_recall, morph_f1,) = self._metrics_dict[split][
            "morph_f1"
        ]._return_stats()

        metrics_dict = {
            f"{split}_lemma_acc": lemma_acc,
            f"{split}_lemma_se": lemma_se,
            f"{split}_morph_tag_acc": morph_tag_acc,
            f"{split}_morph_tag_se": morph_tag_se,
            f"{split}_morph_set_acc": morph_set_acc,
            f"{split}_morph_set_se": morph_set_se,
            f"{split}_morph_precision": morph_precision,
            f"{split}_morph_recall": morph_recall,
            f"{split}_morph_f1": morph_f1,
            f"{split}_lemma_dist": lemma_dist,
            f"{split}_lemma_dist_se": lemma_dist_se,
        }

        self.log_dict(metrics_dict)
        # return metrics_dict

    def training_step(self, batch, batch_idx):

        # For the first epoch, the transformer is not trained
        if self.current_epoch == 0:
            for p in self.transformer.parameters():
                p.requires_grad = False

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            _,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.log_dict(
            {f"train_{k}_loss": v for k, v in losses.items()},
            on_step=True,
            prog_bar=True,
        )
        self.metrics("train", lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log_metrics("train")
        self.clear_metrics("train")

    def validation_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            _,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.log_dict(
            {f"valid_{k}_loss": v for k, v in losses.items()},
            on_step=True,
            prog_bar=True,
        )
        self.metrics("valid", lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_metrics("valid")
        self.clear_metrics("valid")

    def test_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            _,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.log_dict(
            {f"test_{k}_loss": v for k, v in losses.items()},
            on_step=True,
            prog_bar=True,
        )
        self.metrics("test", lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log_metrics("test")
        self.clear_metrics("test")
