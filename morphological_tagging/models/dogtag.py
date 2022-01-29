import warnings
from typing import Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel
import pytorch_lightning as pl

from morphological_tagging.metrics import RunningStats, RunningStatsBatch, RunningF1
from morphological_tagging.models.modules import MultiHeadSequenceAttention, ResidualMLP
from morphological_tagging.models.optim import InvSqrtWithLinearWarmupScheduler
from morphological_tagging.models.functional import break_batch, fuse_batch
from utils.common_operations import label_smooth


class DogTag(pl.LightningModule):
    """A PyTorch Lightning implementation of DogTag.

    """

    def __init__(
        self,
        transformer_type: str,
        transformer_name: str,
        transformer_dropout: float,
        embeddings_dropout: float,
        mha_kwargs: Dict[str, Any],
        mha_lr: float,
        intermediate_hdim: int,
        label_smoothing: float,
        mask_p: float,
        transformer_lrs: Dict[int, float],
        clf_lr: float,
        n_warmup_steps: int,
        optim_kwargs: Dict[str, Any],
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        unfreeze_transformer_epoch: int,
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
        self.mha_kwargs = mha_kwargs
        self.intermediate_hdim = intermediate_hdim

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
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)

        self.chars_embed_dropout = nn.Dropout(p=embeddings_dropout)
        self.token_embed_dropout = nn.Dropout(p=embeddings_dropout)

        # Lemma classification =================================================
        # self._lemma_in_features = (
        #    self.word_rnn_kwargs["h_dim"] + self.c2w_kwargs["out_dim"]
        # )

        self.lemma_mha = MultiHeadSequenceAttention(
            d_in=self.config["hidden_size"],
            d_out=self.intermediate_hdim,
            batch_first=True,
            **mha_kwargs,
        )

        self._lemma_in_features = self.intermediate_hdim

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
        self.morph_mha = MultiHeadSequenceAttention(
            d_in=self.config["hidden_size"],
            d_out=self.intermediate_hdim,
            batch_first=True,
            **mha_kwargs,
        )

        self._morph_in_features = self.intermediate_hdim

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
        self.mha_lr = mha_lr
        self.clf_lr = clf_lr
        self.n_warmup_steps = n_warmup_steps
        self.optim_kwargs = optim_kwargs

        self.unfreeze_transformer_epoch = unfreeze_transformer_epoch

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        self.configure_metrics()

        # Special tokens =======================================================
        self.idx_token_pad = idx_token_pad

    def configure_optimizers(self):

        transformer_lrs = [
            {"params": self.transformer._modules[mod], "lr": lr}
            for mod, lr in self.transformer_lrs
        ]

        transformer_optimizer = AdamW(transformer_lrs, **self.optim_kwargs)

        transformer_scheduler = InvSqrtWithLinearWarmupScheduler(
            transformer_optimizer,
            default_lrs=transformer_lrs,
            n_warmup_steps=self.n_warmup_steps,
        )

        lrs = []
        lrs.append({"params": self.lemma_mha.parameters(), "lr": self.mha_lr})
        lrs.append({"params": self.morph_mha.parameters(), "lr": self.mha_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_unf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_fac.parameters(), "lr": self.clf_lr})

        rest_optimizer = AdamW(lrs, **self.optim_kwargs)

        rest_scheduler = InvSqrtWithLinearWarmupScheduler(
            rest_optimizer, default_lrs=lrs, n_warmup_steps=self.n_warmup_steps
        )

        return (
            [transformer_optimizer, rest_optimizer],
            [
                {"scheduler": transformer_scheduler, "interval": "step"},
                {"scheduler": rest_scheduler, "interval": "step"},
            ],
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
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
        transformer_input = self.tokenizer(
            tokens_raw, is_split_into_words=True, return_tensors="pt", padding=True,
        )

        # Mask characters, while avoiding <PAD> chars
        if self.mask_p >= 0.0 and self.training:
            transformer_input["input_ids"] = torch.where(
                torch.logical_or(
                    torch.bernoulli(
                        transformer_input["input_ids"], 1 - self.mask_p
                    ).bool(),
                    transformer_input["attention_mask"] == 0,
                ),
                transformer_input["input_ids"],
                self.tokenizer.mask_token_id,
            )

        transformer_output = self.transformer(**transformer_input,).last_hidden_state

        transformer_output = self.chars_embed_dropout(transformer_output)

        # Contextualized token embeddings as a [B x L_s, L_t, D] tensor
        cte, cte_mask = break_batch(transformer_output, char_lens)

        # Lemma classification =========================================================
        lemma_h = self.lemma_mha(cte, cte_mask)

        # Move to [B, L_s, D] tensor
        lemma_h = fuse_batch(lemma_h, token_lens)

        lemma_h = self.token_embed_dropout(lemma_h)

        lemma_logits = self.lemma_clf(lemma_h)

        # Morph classification =================================================
        morph_h = self.morph_mha(cte, cte_mask)

        # Move to [B, L_s, D] tensor
        morph_h = fuse_batch(morph_h, token_lens)

        morph_h = self.token_embed_dropout(morph_h)

        morph_logits_unf = self.morph_clf_unf(morph_h)

        if not skip_morph_reg:
            morph_logits_fac = self.morph_clf_fac(morph_h)

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

        if split in self._metrics_dict.keys():
            self._metrics_dict[split] = {
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            }

        else:
            warnings.warn(
                f"{split} is not in the metrics_dict keys. Metrics are not cleared currently."
            )

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

    def on_train_epoch_start(self):
        transformer_scheduler = self.lr_schedulers()[0]
        if self.current_epoch < self.unfreeze_transformer_epoch:
            transformer_scheduler.freeze()
        else:
            transformer_scheduler.thaw()

    def training_step(self, batch, batch_idx, optimizer_idx):

        (
            char_lens,
            _,
            token_lens,
            tokens_raw,
            _,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, token_lens, tokens_raw
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
            {f"train_{k}_loss": v for k, v in losses.items()}, prog_bar=True,
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

        self.log_dict({f"valid_{k}_loss": v for k, v in losses.items()})
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

        self.log_dict({f"test_{k}_loss": v for k, v in losses.items()})
        self.metrics("test", lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log_metrics("test")
        self.clear_metrics("test")
