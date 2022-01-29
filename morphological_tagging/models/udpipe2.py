from typing import Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import pytorch_lightning as pl

from morphological_tagging.metrics import RunningStats, RunningStatsBatch, RunningF1
from morphological_tagging.models.modules import (
    Char2Word,
    ResidualRNN,
    ResidualMLP,
)
from morphological_tagging.models.preprocessor import UDPipe2PreProcessor
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
        preprocessor_kwargs: dict,
        word_rnn_kwargs: dict,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        w_embedding_dim: int = 512,
        pretrained_embedding_dim: int = 300,
        dropout: float = 0.5,
        token_mask_p: float = 0.2,
        label_smoothing: float = 0.03,
        reg_loss_weight: float = 1.0,
        lr: float = 1e-3,
        betas: Tuple[float] = (0.9, 0.99),
        weight_decay=0,
        scheduler_name: Tuple[str, None] = None,
        scheduler_kwargs: Tuple[dict, None] = None,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.c2w_kwargs = c2w_kwargs
        self.w_embedding_dim = w_embedding_dim
        self.pretrained_embedding_dim = pretrained_embedding_dim
        self.word_rnn_kwargs = word_rnn_kwargs

        # Number of classes ====================================================
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.n_morph_cats = n_morph_cats

        # Special tokens =======================================================
        self.unk_token = unk_token
        self.pad_token = pad_token

        # Preprocessor =========================================================
        self.preprocessor_kwargs = preprocessor_kwargs
        self.preprocessor = UDPipe2PreProcessor(**self.preprocessor_kwargs)
        self.preprocessor.freeze_and_eval()

        # Embedding Modules ====================================================
        self.c2w_embedder = Char2Word(
            vocab_len=len(char_vocab),
            padding_idx=char_vocab[pad_token],
            **self.c2w_kwargs,
        )

        self.token_pad_idx = token_vocab[pad_token]

        self.w_embedder = nn.Embedding(
            num_embeddings=len(token_vocab),
            embedding_dim=self.w_embedding_dim,
            padding_idx=self.token_pad_idx,
            sparse=True,
        )

        self._total_embedding_size = (
            self.c2w_kwargs["embedding_dim"]
            + self.w_embedding_dim
            + self.preprocessor.dim
        )

        self.embed_dropout = nn.Dropout(p=dropout)

        # Word-level RNN =======================================================
        self.word_rnn = ResidualRNN(
            input_size=self._total_embedding_size, **self.word_rnn_kwargs,
        )

        # Lemma classification =================================================
        self._lemma_in_features = (
            self.word_rnn_kwargs["h_dim"] + self.c2w_kwargs["out_dim"]
        )

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
        self._morph_in_features = self.word_rnn_kwargs["h_dim"]

        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_tags
            ),
        )

        self.morph_clf_fac = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualMLP(
                        in_features=self._morph_in_features,
                        out_features=self._morph_in_features,
                    ),
                    nn.Linear(in_features=self._morph_in_features, out_features=1),
                )
                for _ in range(self.n_morph_cats)
            ]
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.dropout = dropout

        self.unk_token_idx = token_vocab[unk_token]
        self.token_mask_p = token_mask_p
        self._token_mask = D.bernoulli.Bernoulli(
            torch.tensor([token_mask_p], device=self.device)
        )

        self.label_smoothing = label_smoothing
        self.reg_loss_weight = reg_loss_weight
        self.weight_decay = weight_decay

        # ======================================================================
        # Optimization
        # ======================================================================
        self.lr = lr
        self.betas = betas
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        self.configure_metrics()

    def configure_optimizers(self):

        optimizer_embeddings = optim.SparseAdam(
            [
                {"params": self.w_embedder.parameters()},
                {"params": self.c2w_embedder.embed.parameters()},
            ],
            lr=self.lr,
            betas=self.betas,
        )

        if self.weight_decay > 0.0:
            rest_opt = optim.AdamW
        else:
            rest_opt = optim.Adam

        optimizer_rest = rest_opt(
            [
                *[
                    {"params": p}
                    for n, p in self.c2w_embedder.named_parameters()
                    if (not "embed" in n)
                ],
                {"params": self.word_rnn.parameters()},
                {"params": self.lemma_clf.parameters()},
                {"params": self.morph_clf_unf.parameters()},
                {"params": self.morph_clf_fac.parameters()},
            ],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name is None:
            optimizers = [optimizer_embeddings, optimizer_rest]
            schedulers = None

        elif self.scheduler_name.lower() == "step":
            scheduler_embeddings = MultiStepLR(
                optimizer_embeddings, **self.scheduler_kwargs
            )
            scheduler_rest = MultiStepLR(optimizer_rest, **self.scheduler_kwargs)

            optimizers = [optimizer_embeddings, optimizer_rest]
            schedulers = [scheduler_embeddings, scheduler_rest]

        return optimizers, schedulers

    @property
    def device(self):
        return next(self.parameters()).device

    def _trainable_modules(self):
        return [
            self.c2w_embedder,
            self.w_embedder,
            self.embed_dropout,
            self.word_rnn,
            self.lemma_clf,
            self.morph_clf_unf,
            self.morph_clf_fac,
        ]

    def train(self):
        for mod in self._trainable_modules():
            mod.train()

    def eval(self):
        for mod in self._trainable_modules():
            mod.eval()

    def configure_metrics(self):

        self._metrics_dict = {
            "train": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "valid": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "test": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "predict": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
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
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
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
        losses,
        lemma_logits,
        lemma_tags,
        morph_logits,
        morph_tags,
        token_lens=None,
        tokens_raw=None,
    ):

        self._metrics_dict[split]["loss_total"](losses["total"].detach().cpu().item())
        self._metrics_dict[split]["loss_lemma"](losses["lemma"].detach().cpu().item())
        self._metrics_dict[split]["loss_morph"](losses["morph"].detach().cpu().item())
        self._metrics_dict[split]["loss_morph_reg"](
            losses["morph_reg"].detach().cpu().item()
        )

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

        loss_total, _, loss_total_se, _, _ = self._metrics_dict[split][
            "loss_total"
        ]._return_stats()
        loss_lemma, _, loss_lemma_se, _, _ = self._metrics_dict[split][
            "loss_lemma"
        ]._return_stats()
        loss_morph, _, loss_morph_se, _, _ = self._metrics_dict[split][
            "loss_morph"
        ]._return_stats()
        loss_morph_reg, _, loss_morph_reg_se, _, _ = self._metrics_dict[split][
            "loss_morph_reg"
        ]._return_stats()

        lemma_acc, _, lemma_acc_se = self._metrics_dict[split][
            "lemma_acc"
        ]._return_stats()
        (lemma_dist, _, lemma_dist_se, _, _,) = self._metrics_dict[split][
            "lemma_lev_dist"
        ]._return_stats()

        morph_tag_acc, _, morph_tag_acc_se = self._metrics_dict[split][
            "morph_tag_acc"
        ]._return_stats()
        morph_set_acc, _, morph_set_acc_se = self._metrics_dict[split][
            "morph_set_acc"
        ]._return_stats()
        (morph_precision, morph_recall, morph_f1,) = self._metrics_dict[split][
            "morph_f1"
        ]._return_stats()

        metrics_dict = {
            f"{split}_loss_total": loss_total,
            f"{split}_loss_total_se": loss_total_se,
            f"{split}_loss_lemma": loss_lemma,
            f"{split}_loss_lemma_se": loss_lemma_se,
            f"{split}_loss_morph": loss_morph,
            f"{split}_loss_morph_se": loss_morph_se,
            f"{split}_loss_morph_reg": loss_morph_reg,
            f"{split}_loss_morph_reg_se": loss_morph_reg_se,
            f"{split}_lemma_acc": lemma_acc,
            f"{split}_lemma_acc_se": lemma_acc_se,
            f"{split}_morph_tag_acc": morph_tag_acc,
            f"{split}_morph_tag_acc_se": morph_tag_acc_se,
            f"{split}_morph_set_acc": morph_set_acc,
            f"{split}_morph_set_acc_se": morph_set_acc_se,
            f"{split}_morph_precision": morph_precision,
            f"{split}_morph_recall": morph_recall,
            f"{split}_morph_f1": morph_f1,
            f"{split}_lemma_dist": lemma_dist,
            f"{split}_lemma_dist_se": lemma_dist_se,
        }

        self.log_dict(metrics_dict)

        return metrics_dict

    def _unpack_input(self, batch):

        batch_ = []
        for x in batch:
            if isinstance(x, torch.Tensor) and x.device != self.device:
                batch_.append(x.to(self.device))
            else:
                batch_.append(x)

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch_

        # The lens tensors need to be on CPU in case of packing
        if isinstance(char_lens, list):
            char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

        if isinstance(token_lens, list):
            token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

        return (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        )

    def preprocess(self, token_lens, tokens_raw):
        return self.preprocessor((token_lens, tokens_raw), pre_tokenized=True)

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
        skip_morph_reg: bool = False,
    ) -> Tuple[torch.Tensor]:

        c2w_e = self.c2w_embedder.forward(chars=chars, char_lens=char_lens)

        seqs = []
        beg = torch.tensor([0])
        for l in token_lens:
            seqs.append(c2w_e[beg : beg + l])
            beg += l

        c2w_e = pad_sequence(seqs, padding_value=self.token_pad_idx)

        tokens_swapped = torch.where(
            self._token_mask.sample(tokens.size()).squeeze().bool().to(self.device),
            tokens,
            self.unk_token_idx,
        )

        w_e = self.w_embedder.forward(input=tokens_swapped)

        if pretrained_embeddings is not None:
            e = torch.cat([c2w_e, w_e, pretrained_embeddings], dim=-1)
        else:
            e = torch.cat([c2w_e, w_e], dim=-1)

        e = self.embed_dropout(e)

        h = self.word_rnn(e)

        lemma_logits = self.lemma_clf(torch.cat([h, c2w_e], dim=-1))

        morph_logits_unf = self.morph_clf_unf(h)

        if not skip_morph_reg:
            morph_logits_fac = [fac(h) for fac in self.morph_clf_fac]

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
            ignore_index=-1,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != -1])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = 0
            for i, fac_logits in enumerate(morph_logits_fac):
                cats_target = morph_cats[:, :, i].unsqueeze(-1)

                morph_fac_loss_ = F.binary_cross_entropy_with_logits(
                    fac_logits,
                    label_smooth(self.label_smoothing, cats_target.float()),
                    reduction="none",
                )
                morph_fac_loss_ = torch.mean(morph_fac_loss_[cats_target != -1])

                morph_fac_loss += morph_fac_loss_

            morph_fac_loss /= len(morph_logits_fac)

            loss = lemma_loss + morph_unf_loss + self.reg_loss_weight * morph_fac_loss
            losses = {
                "total": lemma_loss + morph_unf_loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def step(self, batch, split: str = None):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(split, losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "train", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

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
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "valid", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

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
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics("test", losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log_metrics("test")
        self.clear_metrics("test")
