import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from nmt_adapt.optim import InvSqrtWithLinearWarmupScheduler, LinearDecay, DummyScheduler
from nmt_adapt.modules import SequenceMask, TokenClassifier
from utils.common_operations import label_smooth

class TokenDataloader():

    def __init__(self, dataset, max_tokens: int, max_sents: int, length_sort: bool = False, shuffle: bool = True):

        self.dataset = dataset

        self.dataset = self.dataset.map(self.mean_length)

        if length_sort:
            self.dataset = self.dataset.sort("length", reverse=True)

        self.shuffle = shuffle

        self.max_tokens = max_tokens
        self.max_sents = max_sents

        self._batches = self._get_batches()
        self._i = 0

    def _get_batches(self):

        batch_tokens, batch_sents = [0], [0]
        for l in self.dataset["length"]:

            if (batch_tokens[-1] == 0) or \
                (batch_tokens[-1] + l <= self.max_tokens) \
                    and (batch_sents[-1] + 1 <= self.max_sents):
                batch_tokens[-1] += l
                batch_sents[-1] += 1

            else:
                batch_tokens += [l]
                batch_sents += [1]

        return batch_sents

    @staticmethod
    def mean_length(example):
        return {"length": (len(example["src_text"]) + len(example["tgt_tokens"])) / 2}

    def __iter__(self):
        return self

    def __next__(self):

        self._i = 0 if self._i >= len(self._batches) else self._i

        if self._i == 0 and self.shuffle:
            self.dataset = self.dataset.shuffle()
            self._batches = self._get_batches()

        n = sum(self._batches[:self._i])
        batch_size = self._batches[self._i]

        self._i += 1

        return self.dataset[n:n+batch_size]

    def __len__(self):
        return len(self._batches)

class FineTuner(nn.Module):

    def __init__(
        self,
        model_name: str,
        nmt_kwargs: dict,
        optimizer_algorithm: str,
        optimizer_scheduler: typing.Optional[str] = None,
        optimizer_kwargs: typing.Optional[dict] = dict(),
        optimizer_scheduler_kwargs: typing.Optional[dict] = dict(),
        device: typing.Optional[torch.device] = None,
    ):
        super().__init__()

        # ======================================================================
        # Import the base model
        # ======================================================================
        # Marian NMT model =============================================================
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.dropout = nmt_kwargs.get("dropout", 0.1)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            config=self.model_config
        )

        self.seq_mask = SequenceMask(
            mask_p = nmt_kwargs.get("seq_mask", 0.0),
            mask_idx=self.tokenizer.unk_token_id,
        )

        self._eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["eos_token"]]
        )[0]
        self._pad_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["pad_token"]]
        )[0]

        self.model_config = self.model.config

        if device is not None:
            self.model.to(device)

        # ======================================================================
        # Build the optimizers
        # ======================================================================
        if optimizer_algorithm.lower() == "adam":
            self.nmt_optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_kwargs["nmt_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

        elif optimizer_algorithm.lower() == "adamw" or (
            optimizer_algorithm.lower() == "adam"
            and "weight_decay" in optimizer_kwargs.keys()
        ):
            self.nmt_optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_kwargs["nmt_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

        else:
            raise NotImplementedError(
                f"Optimizer {optimizer_algorithm.lower()} not implemented yet."
            )

        if optimizer_scheduler is None:
            self.optimizer_scheduler = DummyScheduler()

        elif optimizer_scheduler.lower() == "linear":

            self.nmt_optimizer_scheduler = LinearDecay(
                self.nmt_optimizer, **optimizer_scheduler_kwargs
            )

        elif optimizer_scheduler.lower() == "inv_sqrt":

            self.nmt_optimizer_scheduler = InvSqrtWithLinearWarmupScheduler(
                self.nmt_optimizer, **optimizer_scheduler_kwargs
            )

        else:
            raise NotImplementedError(
                f"LR scheduler {optimizer_scheduler.lower()} not implemented yet."
            )

        # ======================================================================
        # Additional training details
        # ======================================================================

        self.grad_clip_val = optimizer_kwargs.get("grad_val", torch.inf)
        self.grad_clip_norm = optimizer_kwargs.get("grad_norm", 20)

        self.nmt_label_smoothing = nmt_kwargs.get("label_smoothing", 0)

        # ======================================================================
        # Tracking metrics
        # ======================================================================
        self.epoch = 0
        self.global_step = 0

    def train_step(self, batch: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.Dict]:

        self.train()

        # Tokenize and prep batch ======================================================
        src_text = batch["src_text"]
        ref_tokens = batch["tgt_tokens"]

        src_input = self.tokenizer(src_text, padding=True, truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(
                ref_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True,
            )

        labels = torch.where(
            tgt_input["attention_mask"] != 0,
            tgt_input["input_ids"],
            -100,
        )

        # Forward pass =================================================================
        model_output = self.model.forward(
            input_ids=src_input["input_ids"].to(self.model.device),
            attention_mask=src_input["attention_mask"].to(self.model.device),
            output_hidden_states=True,
            labels=labels.to(self.model.device),
        )

        # NMT loss =====================================================================
        nmt_loss = F.cross_entropy(
            input=model_output.logits.view(-1, self.model_config.vocab_size),
            target=labels.to(self.model.device).view(-1),
            ignore_index=-100,
            label_smoothing=self.nmt_label_smoothing,
        )

        logs = {
            "train/loss": nmt_loss.detach().cpu().item(),
            "train/nmt/loss": nmt_loss.detach().cpu().item(),
            "train/batch_size": labels.size(0),
            "train/seq_lengths": sum([len(ref) for ref in ref_tokens]) / labels.size(0)
        }

        return nmt_loss, logs

    def optimize(self, loss, logs):

        self.nmt_optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_val is not None:
            nn.utils.clip_grad_value_(
                self.parameters(),
                self.grad_clip_val
            )

        total_norm = nn.utils.clip_grad_norm_(
            self.parameters(),
            self.grad_clip_norm
        )


        logs["grad_norm"] = total_norm.cpu().item()
        logs["global_step"] = self.global_step

        self.nmt_optimizer.step()

        self.nmt_optimizer_scheduler.step()

        self.global_step += 1

        logs["lr_scale"] = self.nmt_optimizer_scheduler._get_lr_scale()

        return logs

    @torch.no_grad()
    def eval_step(self, batch, split: str = "valid"):

        self.eval()

        # Tokenize and prep batch ======================================================
        src_text = batch["src_text"]
        ref_tokens = batch["tgt_tokens"]

        src_input = self.tokenizer(src_text, padding=True, truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(
                ref_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True,
            )

        labels = torch.where(
            tgt_input["attention_mask"] != 0,
            tgt_input["input_ids"],
            -100,
        )

        # Forward pass =================================================================
        input_ids = self.seq_mask(src_input["input_ids"], src_input["attention_mask"])

        model_output = self.model.forward(
            input_ids=input_ids.to(self.model.device),
            attention_mask=src_input["attention_mask"].to(self.model.device),
            output_hidden_states=True,
            labels=labels.to(self.model.device),
        )

        # NMT loss =====================================================================
        nmt_loss = F.cross_entropy(
            input=model_output.logits.view(-1, self.model_config.vocab_size),
            target=labels.view(-1).to(self.model.device),
            ignore_index=-100
        )

        logs = {
            f"{split}/loss": nmt_loss.detach().cpu().item(),
            f"{split}/nmt/loss": nmt_loss.detach().cpu().item(),
            f"{split}/batch_size": labels.size(0),
        }

        return logs

class MutliTaskMorphTagTrainer(nn.Module):

    def __init__(
        self,
        model_name: str,
        optimizer_algorithm: str,
        morph_tag_clf_kwargs: dict,
        nmt_kwargs: dict,
        tag_to_int: typing.Dict[str, int],
        loss_weight: float = 0.5,
        checkpoint_path: typing.Optional[str] = None,
        optimizer_scheduler: typing.Optional[str] = None,
        optimizer_kwargs: typing.Optional[dict] = dict(),
        optimizer_scheduler_kwargs: typing.Optional[dict] = dict(),
        device: typing.Optional[torch.device] = None,
    ):
        super().__init__()

        # ======================================================================
        # Import the base model
        # ======================================================================
        # Marian NMT model =============================================================
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if checkpoint_path is None:
            self.model_config = AutoConfig.from_pretrained(model_name)
            self.model_config.dropout = nmt_kwargs.get("dropout", 0.1)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                config=self.model_config
            )

        else:
            print(f"Loading from: {checkpoint_path}")
            self.model_config = AutoConfig.from_pretrained(self.model_name)
            self.model_config.dropout = nmt_kwargs.get("dropout", 0.1)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                config=self.model_config
                )

        self.seq_mask = SequenceMask(
            mask_p = nmt_kwargs.get("seq_mask", 0.0),
            mask_idx=self.tokenizer.unk_token_id,
        )

        self._eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["eos_token"]]
        )[0]
        self._pad_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["pad_token"]]
        )[0]

        # Token classifier =============================================================
        self.token_clf = TokenClassifier(
            in_features=self.model_config.d_model,
            hidden_dim=morph_tag_clf_kwargs["hidden_dim"],
            out_features=len(tag_to_int),
            L=self.model_config.decoder_layers,
            layer_dropout=morph_tag_clf_kwargs["layer_dropout"],
        )

        self.tag_to_int = tag_to_int

        if device is not None:
            self.model.to(device)
            self.token_clf.to(device)

        # ======================================================================
        # Build the optimizers
        # ======================================================================
        if optimizer_algorithm.lower() == "adam":
            self.nmt_optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_kwargs["nmt_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

            self.clf_optimizer = optim.Adam(
                self.token_clf.parameters(),
                lr=optimizer_kwargs["clf_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

        elif optimizer_algorithm.lower() == "adamw" or (
            optimizer_algorithm.lower() == "adam"
            and "weight_decay" in optimizer_kwargs.keys()
        ):
            self.nmt_optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_kwargs["nmt_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

            self.clf_optimizer = optim.AdamW(
                self.token_clf.parameters(),
                lr=optimizer_kwargs["clf_lr"],
                betas=tuple(optimizer_kwargs.get("betas", (0.9, 0.999))),
                weight_decay=optimizer_kwargs.get("weight_decay", 0),
            )

        else:
            raise NotImplementedError(
                f"Optimizer {optimizer_algorithm.lower()} not implemented yet."
            )

        if optimizer_scheduler is None:
            self.optimizer_scheduler = DummyScheduler()

        elif optimizer_scheduler.lower() == "linear":

            self.nmt_optimizer_scheduler = LinearDecay(
                self.nmt_optimizer, **optimizer_scheduler_kwargs
            )

            self.clf_optimizer_scheduler = InvSqrtWithLinearWarmupScheduler(
                self.clf_optimizer, 0.04 * optimizer_scheduler_kwargs["n_warmup_steps"]
            )

        elif optimizer_scheduler.lower() == "inv_sqrt":

            self.nmt_optimizer_scheduler = InvSqrtWithLinearWarmupScheduler(
                self.nmt_optimizer, **optimizer_scheduler_kwargs
            )

            self.clf_optimizer_scheduler = InvSqrtWithLinearWarmupScheduler(
                self.clf_optimizer, **optimizer_scheduler_kwargs
            )

        else:
            raise NotImplementedError(
                f"LR scheduler {optimizer_scheduler.lower()} not implemented yet."
            )

        # ======================================================================
        # Additional training details
        # ======================================================================

        self.grad_clip_val = optimizer_kwargs.get("grad_val", torch.inf)
        self.grad_clip_norm = optimizer_kwargs.get("grad_norm", 20)

        self.nmt_label_smoothing = nmt_kwargs.get("label_smoothing", 0)
        self.morph_tag_label_smoothing = morph_tag_clf_kwargs.get("label_smoothing", 0)

        self.loss_weight = loss_weight

        # ======================================================================
        # Tracking metrics
        # ======================================================================
        self.epoch = 0
        self.global_step = 0

    def _tag_set_to_tensor(self, tag_set):

        tags_tensor = torch.zeros(len(self.tag_to_int))

        tags = [
            self.tag_to_int.get(tag, len(self.tag_to_int))
            for tag in tag_set
            ]

        tags_tensor[tags] = 1

        return tags_tensor

    def train_step(self, batch: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.Dict]:

        self.model.train()
        self.token_clf.train()

        # Tokenize and prep batch ======================================================
        src_text = batch["src_text"]
        ref_tokens = batch["tgt_tokens"]

        src_input = self.tokenizer(src_text, padding=True, truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(
                ref_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True,
            )

        labels = torch.where(
            tgt_input["attention_mask"] != 0,
            tgt_input["input_ids"],
            -100,
        )

        morph_tag_label_tensor = pad_sequence([
            torch.stack(
                [
                    self._tag_set_to_tensor(tag_set) for tag_set in tag_seq
                    ],
                dim=0
                )
            for tag_seq in batch["morph_tags"]
            ],
            batch_first=True,
            padding_value=-100
        )

        # Forward pass =================================================================
        input_ids = self.seq_mask(src_input["input_ids"], src_input["attention_mask"])

        model_output = self.model.forward(
            input_ids=input_ids.to(self.model.device),
            attention_mask=src_input["attention_mask"].to(self.model.device),
            output_hidden_states=True,
            labels=labels.to(self.model.device),
        )

        # NMT loss =====================================================================
        nmt_loss = F.cross_entropy(
            input=model_output.logits.view(-1, self.model_config.vocab_size),
            target=labels.to(self.model.device).view(-1),
            ignore_index=-100,
            label_smoothing=self.nmt_label_smoothing,
        )

        # Token Classifier loss =====================================================================
        decoder_hidden_states = torch.stack(model_output.decoder_hidden_states, dim=-2)

        with self.tokenizer.as_target_tokenizer():
            token_logits = self.token_clf(
                decoder_hidden_states,
                tgt_input["input_ids"].to(self.model.device),
                self.tokenizer,
            )

        morph_tag_label_tensor_ = morph_tag_label_tensor[:, :token_logits.size(1), :].to(self.model.device)

        morph_tags_loss = F.binary_cross_entropy_with_logits(
            input=token_logits,
            target=label_smooth(
                epsilon=self.morph_tag_label_smoothing,
                labels=morph_tag_label_tensor_,
                K=morph_tag_label_tensor.size(-1)
                ),
            reduction='none',
        )

        morph_tags_loss = torch.mean(morph_tags_loss[morph_tag_label_tensor_ != -100])

        logs = {
            "train/loss": nmt_loss.detach().cpu().item() + morph_tags_loss.detach().cpu().item(),
            "train/nmt/loss": nmt_loss.detach().cpu().item(),
            "train/morph_tag/loss": morph_tags_loss.detach().cpu().item(),
            "train/batch_size": labels.size(0),
            "train/seq_lengths": sum([len(ref) for ref in ref_tokens]) / labels.size(0)
        }

        return (1-self.loss_weight) * nmt_loss + self.loss_weight * morph_tags_loss, logs

    def optimize(self, loss, logs):

        self.nmt_optimizer.zero_grad()
        self.clf_optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_val is not None:
            nn.utils.clip_grad_value_(
                self.parameters(),
                self.grad_clip_val
            )

        total_norm = nn.utils.clip_grad_norm_(
            self.parameters(),
            self.grad_clip_norm
        )

        logs["grad_norm"] = total_norm.cpu().item()
        logs["global_step"] = self.global_step

        self.nmt_optimizer.step()
        self.clf_optimizer.step()

        self.nmt_optimizer_scheduler.step()
        self.clf_optimizer_scheduler.step()

        self.global_step += 1

        logs["lr_scale"] = self.nmt_optimizer_scheduler._get_lr_scale()

        return logs

    @torch.no_grad()
    def eval_step(self, batch, split: str = "valid"):

        self.model.eval()

        # Tokenize and prep batch ======================================================
        src_text = batch["src_text"]
        ref_tokens = batch["tgt_tokens"]

        src_input = self.tokenizer(src_text, padding=True, truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(
                ref_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True,
            )

        labels = torch.where(
            tgt_input["attention_mask"] != 0,
            tgt_input["input_ids"],
            -100,
        )

        morph_tag_label_tensor = pad_sequence([
            torch.stack(
                [
                    self._tag_set_to_tensor(tag_set) for tag_set in tag_seq
                    ],
                dim=0
                )
            for tag_seq in batch["morph_tags"]
            ],
            batch_first=True,
            padding_value=-100
        )

        # Forward pass =================================================================
        model_output = self.model.forward(
            input_ids=src_input["input_ids"].to(self.model.device),
            attention_mask=src_input["attention_mask"].to(self.model.device),
            output_hidden_states=True,
            labels=labels.to(self.model.device),
        )

        # NMT loss =====================================================================
        nmt_loss = F.cross_entropy(
            input=model_output.logits.view(-1, self.model_config.vocab_size),
            target=labels.view(-1).to(self.model.device),
            ignore_index=-100
        )

        # Token Classifier loss =====================================================================
        decoder_hidden_states = torch.stack(model_output.decoder_hidden_states, dim=-2)

        with self.tokenizer.as_target_tokenizer():
            token_logits = self.token_clf(
                decoder_hidden_states,
                tgt_input["input_ids"],
                self.tokenizer,
            )

        morph_tag_label_tensor_ = morph_tag_label_tensor[:, :token_logits.size(1), :].to(self.model.device)

        morph_tags_loss = F.binary_cross_entropy_with_logits(
            input=token_logits,
            target=label_smooth(
                epsilon=self.morph_tag_label_smoothing,
                labels=morph_tag_label_tensor_,
                K=morph_tag_label_tensor.size(-1)
                ),
            reduction='none',
        )

        morph_tags_loss = torch.mean(morph_tags_loss[morph_tag_label_tensor_ != -100])

        # Morph tag metrics ========================================================
        morph_tags_pred = torch.round(torch.sigmoid(token_logits)).cpu()
        morph_tags_gt = morph_tag_label_tensor[:, :token_logits.size(1), :]
        mask = torch.any(morph_tags_gt != -100, dim=-1)

        token_acc = torch.all(morph_tags_pred == morph_tags_gt, dim=2).float()

        token_acc_batch = torch.mean(token_acc[mask])

        union = torch.sum(torch.logical_and(morph_tags_pred, morph_tags_gt), dim=-1)

        pred_positives = torch.sum(morph_tags_pred, dim=-1)
        true_positives = torch.sum(morph_tags_gt, dim=-1)

        precision = union / pred_positives
        recall = union / true_positives

        f1 = torch.nan_to_num(2 * precision * recall / (precision + recall), 0)

        f1_batch = torch.mean(f1[mask])

        logs = {
            f"{split}/loss": nmt_loss.detach().cpu().item() + morph_tags_loss.detach().cpu().item(),
            f"{split}/nmt/loss": nmt_loss.detach().cpu().item(),
            f"{split}/morph_tag/loss": morph_tags_loss.detach().cpu().item(),
            f"{split}/morph_tag/token_acc": token_acc_batch.cpu().item(),
            f"{split}/morph_tag/f1": f1_batch.cpu().item(),
            f"{split}/batch_size": labels.size(0),
        }

        return logs
