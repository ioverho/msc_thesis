from typing import Tuple, Union

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
from utils.common_operations import label_smooth
from utils.experiment import Timer


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
        scheduler_name: Tuple[str, None] = None,
        scheduler_kwargs: Tuple[dict, None] = None,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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
            + self.pretrained_embedding_dim
        )

        self.embed_dropout = nn.Dropout(p=dropout)

        # Word-level RNN =======================================================
        self.word_rnn = ResidualRNN(
            input_size=self._total_embedding_size,
            batch_first=False,
            dropout=dropout,
            **self.word_rnn_kwargs,
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
        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(in_features=512, out_features=512),
            nn.Linear(in_features=512, out_features=self.n_morph_tags - 1),
        )

        self.morph_clf_fac = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualMLP(in_features=512, out_features=512),
                    nn.Linear(in_features=512, out_features=1),
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

    def configure_optimizers(self):

        optimizer_embeddings = optim.SparseAdam(
            self.w_embedder.parameters(), lr=self.lr, betas=self.betas
        )
        optimizer_rest = optim.Adam(
            [
                {"params": self.c2w_embedder.parameters()},
                {"params": self.word_rnn.parameters()},
                {"params": self.lemma_clf.parameters()},
                {"params": self.morph_clf_unf.parameters()},
                {"params": self.morph_clf_fac.parameters()},
            ],
            lr=self.lr,
            betas=self.betas,
        )

        if self.scheduler_name is None:
            return [optimizer_embeddings, optimizer_rest]

        elif self.scheduler_name.lower() == "step":
            scheduler_embeddings = MultiStepLR(
                optimizer_embeddings, **self.scheduler_kwargs
            )
            scheduler_rest = MultiStepLR(optimizer_rest, **self.scheduler_kwargs)

            return (
                [optimizer_embeddings, optimizer_rest],
                [scheduler_embeddings, scheduler_rest],
            )

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
        skip_morph_reg: bool = False
    ) -> Tuple[torch.Tensor]:

        # The lens tensors need to be on CPU in case of packing
        if isinstance(char_lens, list):
            char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

        if isinstance(token_lens, list):
            token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

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

        e = torch.cat([c2w_e, w_e, pretrained_embeddings], dim=-1)

        e = self.embed_dropout(e)

        h = self.word_rnn(e)

        lemma_logits = self.lemma_clf(torch.cat([h, c2w_e], dim=-1))

        morph_logits_unf = self.morph_clf_unf(h)

        if (not skip_morph_reg):
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

        self.lemma_acc_epoch = RunningStatsBatch()
        self.lemma_lev_dist = RunningStats()

        self.morph_tag_acc_epoch = RunningStatsBatch()
        self.morph_set_acc_epoch = RunningStatsBatch()
        self.morph_f1_epoch = RunningF1()

    @torch.no_grad()
    def metrics(self, lemma_logits, lemma_tags, morph_logits, morph_tags, token_lens = None, tokens_raw  = None):

        if (token_lens is not None) and (tokens_raw is not None):
            skip_lev_dist = False
        else:
            skip_lev_dist = True

        ## Lemma CLF metrics
        lemma_preds = torch.argmax(lemma_logits, dim=-1).permute(1,0).detach().cpu().numpy()
        lemma_targets = lemma_tags.permute(1,0).detach().cpu().numpy()
        lemma_mask = np.where((lemma_tags != -1).permute(1,0).detach().cpu().numpy(), 1., np.nan)

        if not skip_lev_dist:
            # TODO (ivo): implement lev_distance reporting inside model
            raise NotImplementedError
            for i, (preds_seq, target_seq, seq_len) in enumerate(zip(lemma_preds, lemma_targets, token_lens)):
                for pred, target, token in zip(preds_seq[:seq_len], target_seq[:seq_len], tokens_raw[i]):
                    pred_lemma_script = corpus.id_to_script[pred]
                    pred_lemma = apply_lemma_script(token, pred_lemma_script)

                    target_lemma_script = corpus.id_to_script[target]
                    target_lemma = apply_lemma_script(token, target_lemma_script)

                    lemma_lev_dist(distance(pred_lemma, target_lemma), output=False)

        self.lemma_acc_epoch(lemma_preds == lemma_targets, lemma_mask)

        ## Morph. CLF metrics
        morph_preds = torch.round(torch.sigmoid(morph_logits)).permute(1,0,2).detach().cpu().numpy()
        morph_targets = morph_tags.permute(1,0,2).detach().cpu().numpy()
        morph_mask = np.where((morph_tags != -1).permute(1,0,2).detach().cpu().numpy(), 1., np.nan)
        morph_set_mask = np.max(morph_mask, axis=-1)

        item_match = (morph_preds == morph_targets)
        set_match = np.all((morph_preds == morph_targets), axis=-1)

        # Morph. Acc
        self.morph_tag_acc_epoch(item_match, morph_mask)
        self.morph_set_acc_epoch(set_match, morph_set_mask)

        self.morph_f1_epoch(morph_preds, morph_targets, morph_set_mask)

    def log_metrics(self, split):

        try:
            lemma_acc, _, lemma_se = self.lemma_acc_epoch._return_stats()
            lemma_dist, _, lemma_dist_se, lemma_dist_min, lemma_dist_max = self.lemma_lev_dist._return_stats()

            morph_tag_acc, _, morph_tag_se = self.morph_tag_acc_epoch._return_stats()
            morph_set_acc, _, morph_set_se = self.morph_set_acc_epoch._return_stats()
            morph_precision, morph_recall, morph_f1 = self.morph_f1_epoch._return_stats()

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
                f"{split}_lemma_dist_min": lemma_dist_min,
                f"{split}_lemma_dist_max": lemma_dist_max,
            }

            self.log_dict(metrics_dict)

        except ValueError:
            return None

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        self.configure_metrics()

    def training_step(self, batch, batch_idx, optimizer_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch[0]

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

        self.log_dict({f"train_{k}_loss": v for k, v in losses.items()}, on_step=True, prog_bar=True)
        self.metrics(lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        self.log_metrics("train")

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.configure_metrics()

    def validation_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

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

        self.log_dict({f"valid_{k}_loss": v for k, v in losses.items()}, on_epoch=True)
        self.metrics(lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        self.log_metrics("validation")

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()

        self.configure_metrics()

    def test_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens,
            pretrained_embeddings,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = batch

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

        self.log_dict({f"test_{k}_loss": v for k, v in losses.items()}, on_epoch=True)
        self.metrics(lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        self.log_metrics("test")
