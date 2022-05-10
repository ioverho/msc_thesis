import typing
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nmt_adapt.optim import DummyScheduler, InvSqrtWithLinearWarmupScheduler
from nmt_adapt.gbml import MarianMAMLpp, MarianAnil
from utils.errors import ConfigurationError


class MetaDataLoader(object):
    """Class that controls the task sampling, episode generation and data loading.

    The function takes in a dataset, an index, a task sampler, a tokenizer, the number of lemmas per
    task, the number of samples per lemma, a loading mode, the probability of using the full NMT objective,
    and the probability of uninformed sampling.

    Args:
        dataset:
        index: a dictionary mapping lemmas to their indices in the dataset
        task_sampler: a function that takes a list of tasks and returns a task
        tokenizer: a tokenizer object
        n_lemmas_per_task: int,
        n_samples_per_lemma: int,
        mode: str = "cross_transfer", the episode generation mode. Defaults to cross_transfer
        p_full_nmt: float = 0.0,
        p_uninformed: float = 0.0,
        eval_uninformed: bool = False, whether or not to evaluate using uniform task distribution
    """

    def __init__(
        self,
        dataset,
        index,
        tokenizer,
        task_sampler,
        n_lemmas_per_task: int = 2,
        n_samples_per_lemma: int = 1,
        mode: str = "cross_transfer",
        p_full_nmt: float = 0.0,
        p_uninformed: float = 0.0,
        eval_uninformed: bool = False,
    ):

        self.dataset = dataset
        self.index = index
        self.task_sampler = task_sampler
        self.tokenizer = tokenizer

        if self.n_lemmas_per_task < 2 and mode == "cross_transfer":
            raise ValueError("Number of lemmas per task must be at least 2 for cross_transfer.")
        else:
            self.n_lemmas_per_task = n_lemmas_per_task
        self.n_samples_per_lemma = n_samples_per_lemma

        if mode in {"cross_transfer"}:
            self.mode = mode
        else:
            raise ConfigurationError(f"Dataloader mode {mode} not implemented.")

        self.p_full_nmt = p_full_nmt
        self.p_uninformed = p_uninformed

        self.eval_uninformed = eval_uninformed

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __next__(self):

        if self.training:
            return self._train_batch()

        else:
            return self._eval_batch()

    def _cross_transfer_batches(self, t1, t2):

        # Get the lemmas at the intersection of the tasks
        joint_lemma_set = self.task_sampler.lemma_intersection[t1, t2]

        # Limit the number of lemmas sampled per task to
        # the nearest multiple of 2 <= the set size
        max_allowed_lemmas_total = 2 * (
            len(self.task_sampler.lemma_intersection[t1, t2]) // 2
        )

        n_lemmas_total = min(self.n_lemmas_per_task, max_allowed_lemmas_total)

        # Sample uniformly from the possible lemmas
        sampled_lemmas = random.sample(list(joint_lemma_set), k=n_lemmas_total)

        # Randomly shuffle the tasks
        tasks = [t1, t2] if random.random() < 0.5 else [t2, t1]

        # Get all possible instances of a task/lemma combination
        task_lemma_instances = defaultdict(lambda: defaultdict(list))
        for t in tasks:
            for l in sampled_lemmas:
                task_lemma_instances[t][l] = self.index[t][l]

        # Limit the number of examples per lemma to the minimum size
        # of instances possible for a task
        n_samples_per_lemma_ = min(
            self.n_samples_per_lemma,
            min(len(vv) for v in task_lemma_instances.values() for vv in v.values()),
        )

        # Build the support and query batches
        support_batch, query_batch = [], []
        for i, l in enumerate(sampled_lemmas):
            if i < n_lemmas_total // 2:
                support_batch.extend(
                    random.sample(
                        task_lemma_instances[tasks[0]][l], k=n_samples_per_lemma_
                    )
                )
                query_batch.extend(
                    random.sample(
                        task_lemma_instances[tasks[1]][l], k=n_samples_per_lemma_
                    )
                )

            else:
                support_batch.extend(
                    random.sample(
                        task_lemma_instances[tasks[1]][l], k=n_samples_per_lemma_
                    )
                )
                query_batch.extend(
                    random.sample(
                        task_lemma_instances[tasks[0]][l], k=n_samples_per_lemma_
                    )
                )

        return support_batch, query_batch

    def get_input(self, batch_ids):

        sent_ids, tok_ids = list(map(list, zip(*batch_ids)))

        batch = self.dataset[sent_ids]
        batch["token_id"] = tok_ids

        return batch

    def _collate_batches(self, batch, full_nmt):

        # Get the source text (str)
        source_inputs = self.tokenizer(
            batch["src_text"], return_tensors="pt", padding=True, truncation=True
        )

        if full_nmt:
            # Use conditional LM objective on all words
            # Get the target tokens (List[str]) and token_id as `t` (List[int])
            with self.tokenizer.as_target_tokenizer():
                target_inputs = self.tokenizer(
                    [[self.tokenizer.pad_token] + seq for seq in batch["tgt_tokens"]],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    is_split_into_words=True,
                )

            # Generate labels from target inputs
            # i.e. use next bpe as label whenever it is not padding
            labels = torch.where(
                target_inputs["attention_mask"][:, 1:] == 0,
                -100,
                target_inputs["input_ids"][:, 1:],
            )

        else:
            # Use conditional LM objective on only the target word
            # Feeds in prefix as labels, but masks these during loss computation

            target_prefix = [
                [self.tokenizer.pad_token] + seq[: t + 1]
                for seq, t in zip(batch["tgt_tokens"], batch["token_id"])
            ]
            target_label = [
                seq[t : t + 1] for seq, t in zip(batch["tgt_tokens"], batch["token_id"])
            ]

            with self.tokenizer.as_target_tokenizer():
                # Left truncate prefix string
                target_inputs = self.tokenizer(target_prefix, is_split_into_words=True,)

                target_inputs = {
                    k: pad_sequence(
                        [
                            torch.tensor(seq[-self.tokenizer.model_max_length :])
                            for seq in v
                        ],
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id
                        if k == "input_ids"
                        else 0,
                    )
                    for k, v in target_inputs.items()
                }

                # Get label spe lengths
                spe_lens = torch.tensor(
                    [
                        len(spe_seq) - 1
                        for spe_seq in self.tokenizer(
                            target_label, is_split_into_words=True
                        )["input_ids"]
                    ]
                )

            # Find where the target spe sequences end (tokenizer adds </s> token)
            rows, seq_end = torch.where(target_inputs["input_ids"] == 0)

            # Repeat the rows l times, where l is the length of the target spe sequence for that row
            row_idx = torch.repeat_interleave(rows, spe_lens)

            # Create a tensor that counts from the starting index of the target spe sequence to its end
            seq_idx = torch.repeat_interleave(
                seq_end - spe_lens, spe_lens
            ) + torch.tensor([i for l in spe_lens for i in range(l.item())])

            # Example:
            # target spe sequence in row 0 is of length 3 and ends at index 10
            # target spe sequence in row 1 is of length 2 and ends at index 3
            # row_idx = [0, 0, 0, 1, 1, ...]
            # seq_idx = [7, 8, 9, 1, 2, ...]

            # Use the row and column indices to fill default labels with true labels
            labels = torch.full_like(target_inputs["input_ids"], fill_value=-100)
            labels[row_idx, seq_idx] = target_inputs["input_ids"][row_idx, seq_idx]

            # Offset the tensor by one to skip <pad> token in beginning
            labels = labels[:, 1:]

        return source_inputs, target_inputs, labels

    def _train_batch(self):
        p_obj = random.random()

        if p_obj <= self.p_full_nmt or self.task_sampler is None:

            full_nmt = True
            objective = "full"

            max_batch_size = self.n_lemmas_per_task * self.n_samples_per_lemma

            batches = random.sample(range(len(self.dataset)), k=2 * max_batch_size)

            batches = list(map(lambda x: (x, 0), batches))

            support_batch, query_batch = (
                batches[:max_batch_size],
                batches[max_batch_size:],
            )

        elif p_obj <= self.p_full_nmt + self.p_uninformed:

            full_nmt = False
            objective = "uniform_partial"

            t1, t2 = self.task_sampler.sample_tasks(informed=False)

            support_batch, query_batch = self._cross_transfer_batches(t1, t2)

        else:

            full_nmt = False
            objective = "informed_partial"

            if self.mode == "cross_transfer":

                # Sample from task sampler
                t1, t2 = self.task_sampler.sample_tasks(informed=True)

                support_batch, query_batch = self._cross_transfer_batches(t1, t2)

        support_batch = self.get_input(support_batch)
        query_batch = self.get_input(query_batch)

        support_batch = self._collate_batches(support_batch, full_nmt=full_nmt)
        query_batch = self._collate_batches(query_batch, full_nmt=full_nmt)

        return support_batch, query_batch, objective

    def _eval_batch(self):

        # Sample from task sampler
        t1, t2 = self.task_sampler.sample_tasks(informed=(not self.eval_uninformed))

        support_batch, query_batch = self._cross_transfer_batches(t1, t2)

        support_batch = self.get_input(support_batch)
        query_batch = self.get_input(query_batch)

        supp_src_inputs, supp_tgt_inputs, supp_labels = self._collate_batches(
            support_batch, full_nmt=True
        )
        _, supp_tgt_inputs_part, supp_labels_part = self._collate_batches(
            support_batch, full_nmt=False
        )

        query_src_inputs, query_tgt_inputs, query_labels = self._collate_batches(
            query_batch, full_nmt=True
        )
        _, query_tgt_inputs_part, query_labels_part = self._collate_batches(
            query_batch, full_nmt=False
        )

        return (
            [
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                supp_tgt_inputs_part,
                supp_labels_part,
            ],
            [
                query_src_inputs,
                query_tgt_inputs,
                query_labels,
                query_tgt_inputs_part,
                query_labels_part,
            ],
        )


class MetaTrainer(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        model_name: str,
        meta_learner_algorithm: str,
        inner_lr: float,
        meta_optimizer_algorithm: str,
        meta_lr: float,
        meta_optimizer_scheduler: typing.Optional[str] = None,
        train_meta_batchsize: int = 1,
        valid_meta_batchsize: typing.Optional[int] = None,
        train_k: int = 1,
        valid_k: typing.Optional[int] = None,
        first_order_epochs: int = 0,
        meta_learner_kwargs: typing.Optional[dict] = dict(),
        meta_optimizer_kwargs: typing.Optional[dict] = dict(),
        meta_optimizer_scheduler_kwargs: typing.Optional[dict] = dict(),
        grad_clip_val: typing.Optional[float] = None,
        device: typing.Optional[torch.device] = None,
    ):

        # ======================================================================
        # Import the base model
        # ======================================================================
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, enable_sampling=False
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self._eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["eos_token"]]
        )[0]
        self._pad_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.special_tokens_map["pad_token"]]
        )[0]

        # ======================================================================
        # Build the meta learning algorithm
        # ======================================================================
        if (
            meta_learner_algorithm.lower() == "maml++"
            or meta_learner_algorithm.lower() == "mamlpp"
        ):
            self.meta_learner = MarianMAMLpp(
                self.model, lr=inner_lr, **meta_learner_kwargs
            )
            self._anil = False
        elif (
            meta_learner_algorithm.lower() == "anil"
            or meta_learner_algorithm.lower() == "anil++"
        ):
            self.meta_learner = MarianAnil(
                self.model, lr=inner_lr, **meta_learner_kwargs
            )
            self._anil = True
        else:
            raise NotImplementedError(
                f"Meta learner {meta_learner_algorithm.lower()} not implemented yet."
            )

        if device is not None:
            self.meta_learner.to(device)

        # ======================================================================
        # Build the meta learning outer loop optimizers
        # ======================================================================
        if meta_optimizer_algorithm.lower() == "adam":
            self.meta_optimizer = optim.Adam(
                self.meta_learner.parameters(), lr=meta_lr, **meta_optimizer_kwargs
            )
        elif meta_optimizer_algorithm.lower() == "adamw" or (
            meta_optimizer_algorithm.lower() == "adam"
            and "weight_decay" in meta_optimizer_kwargs.keys()
        ):
            self.meta_optimizer = optim.AdamW(
                self.meta_learner.parameters(), lr=meta_lr, **meta_optimizer_kwargs
            )
        else:
            raise NotImplementedError(
                f"Meta optimizer {meta_optimizer_algorithm.lower()} not implemented yet."
            )

        if meta_optimizer_scheduler is None:
            self.meta_optimizer_scheduler = DummyScheduler()
        elif meta_optimizer_scheduler.lower() == "inv_sqrt":
            self.meta_optimizer_scheduler = InvSqrtWithLinearWarmupScheduler(
                self.meta_optimizer, **meta_optimizer_scheduler_kwargs
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {meta_optimizer_scheduler.lower()} not implemented yet."
            )
        # ======================================================================
        # Additional training details
        # ======================================================================
        self.train_meta_batchsize = train_meta_batchsize
        self.valid_meta_batchsize = (
            valid_meta_batchsize
            if valid_meta_batchsize is not None
            else train_meta_batchsize
        )
        self.train_k = train_k
        self.valid_k = valid_k if valid_k is not None else train_k
        self.first_order_epochs = first_order_epochs

        self.grad_clip_val = grad_clip_val

        # ======================================================================
        # Tracking metrics
        # ======================================================================
        self.epoch = 0
        self.global_step = 0

    def gmbl_step(
        self,
        model,
        src_inputs,
        tgt_inputs,
        labels,
        adapt_steps: int = 0,
        first_order: bool = False,
    ):

        init_loss = 0.0
        loss = 0.0

        if self._anil:
            with torch.no_grad():
                features = model.extract_features(
                    input_ids=src_inputs["input_ids"],
                    attention_mask=src_inputs["attention_mask"],
                    decoder_input_ids=tgt_inputs["input_ids"][:, :-1],
                    decoder_attention_mask=tgt_inputs["attention_mask"][:, :-1],
                )

        if adapt_steps > 0:
            for k in range(adapt_steps):

                if not self._anil:
                    features = model.extract_features(
                        input_ids=src_inputs["input_ids"],
                        attention_mask=src_inputs["attention_mask"],
                        decoder_input_ids=tgt_inputs["input_ids"][:, :-1],
                        decoder_attention_mask=tgt_inputs["attention_mask"][:, :-1],
                    )

                logits = model.classify(features)

                loss = F.cross_entropy(
                    input=logits.permute(0, 2, 1), target=labels.to(logits.device)
                )

                model.adapt(loss, first_order=first_order)

                if k == 0:
                    init_loss = loss.detach().cpu()

        else:
            if not self._anil:
                features = model.extract_features(
                    input_ids=src_inputs["input_ids"],
                    attention_mask=src_inputs["attention_mask"],
                    decoder_input_ids=tgt_inputs["input_ids"][:, :-1],
                    decoder_attention_mask=tgt_inputs["attention_mask"][:, :-1],
                )

            logits = model.classify(features)

            loss = F.cross_entropy(
                input=logits.permute(0, 2, 1), target=labels.to(logits.device)
            )

            init_loss = loss.detach().cpu()

        return model, loss, init_loss

    @staticmethod
    def train_step_metrics(logs):

        supp_pre_loss = torch.stack(logs["supp_pre_loss"])
        query_post_loss = torch.stack(logs["query_post_loss"])

        logs_ = dict()

        for obj in ["full", "uniform_partial", "informed_partial"]:
            obj_mask = [o == obj for o in logs["objective"]]

            if sum(obj_mask) == 0:
                continue

            supp_pre_loss_masked = supp_pre_loss[obj_mask]
            query_post_loss_masked = query_post_loss[obj_mask]

            logs_ |= {
                f"train/supp_loss/{obj}": torch.mean(supp_pre_loss_masked),
                f"train/query_loss/{obj}": torch.mean(query_post_loss_masked),
                f"train/cross_imprv_mor/{obj}": torch.mean(
                    1 - query_post_loss_masked / supp_pre_loss_masked
                ),
                f"train/cross_imprv_rom/{obj}": 1
                - torch.mean(query_post_loss_masked) / torch.mean(supp_pre_loss_masked),
            }

        logs_["train/first_order"] = logs["first_order"].float()

        return logs_

    def train_step(self, data_loader):

        self.meta_learner.train()

        logs = defaultdict(list)

        meta_batch_loss = torch.tensor(0.0, device=self.meta_learner.device)
        for _ in range(self.train_meta_batchsize):

            # Get data
            data_loader.train()
            support_batch, query_batch, objective = next(data_loader)

            supp_src_inputs, supp_tgt_inputs, supp_labels = support_batch
            query_src_inputs, query_tgt_inputs, query_labels = query_batch

            # Clone the model for task-specific adaptation
            first_order = self.first_order_epochs < self.epoch
            task_model = self.meta_learner.clone()

            # ==================================================================
            # Support / Inner loop optimization
            # ==================================================================

            task_model, _, init_loss = self.gmbl_step(
                task_model,
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                adapt_steps=self.train_k,
                first_order=first_order,
            )

            logs["supp_pre_loss"] += [init_loss]

            # ==================================================================
            # Query / Outer loop optimization
            # ==================================================================

            task_model, loss, init_loss = self.gmbl_step(
                task_model,
                query_src_inputs,
                query_tgt_inputs,
                query_labels,
                adapt_steps=0,
                first_order=first_order,
            )

            meta_batch_loss += loss

            logs["query_post_loss"] += [init_loss]

            logs["objective"] += [objective]

        logs = dict(logs)
        logs["first_order"] = torch.tensor(first_order)

        logs = self.train_step_metrics(logs)

        return meta_batch_loss, logs

    def optimize(self, loss, logs):

        self.meta_optimizer.zero_grad()
        loss.backward()

        for p in self.meta_learner.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / self.train_meta_batchsize)

        if self.grad_clip_val is not None:
            nn.utils.clip_grad_value_(
                self.meta_learner.parameters(), self.grad_clip_val
            )

        total_norm = 0.0
        for p in self.meta_learner.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        logs["global_step"] = self.global_step
        logs["grad_norm"] = total_norm

        self.meta_optimizer.step()
        self.meta_optimizer_scheduler.step()

        self.global_step += 1

        logs["lr_scale"] = self.meta_optimizer_scheduler._get_lr_scale()

        logs.update(
            {
                f"inner_lr/{k}": v.detach().data.item()
                for k, v in self.meta_learner.lrs.items()
            }
        )

        return logs

    def eval_step(
        self, data_loader, split: str = "valid", batch_size: typing.Optional[int] = None
    ):

        @torch.no_grad()
        def get_rel_nrl_loss(model, src_inputs, tgt_inputs, labels, labels_part):
            """Computes the NMT loss of a model (adapted or not) on the entire sentence,
            returning losses for both the relevant tokens, and the non-relevant tokens.

            Args:
                model (_type_): _description_
                src_inputs (_type_): _description_
                tgt_inputs (_type_): _description_
                labels (_type_): _description_
                labels_part (_type_): _description_
            """
            # Compute loss of query set prior to adaptation
            # Computes for both relevant and non-relevant tokens
            features = model.extract_features(
                input_ids=src_inputs["input_ids"],
                attention_mask=src_inputs["attention_mask"],
                decoder_input_ids=tgt_inputs["input_ids"][:, :-1],
                decoder_attention_mask=tgt_inputs["attention_mask"][:, :-1],
            )

            logits = model.classify(features)

            loss = F.cross_entropy(
                input=logits.permute(0, 2, 1),
                target=labels.to(logits.device),
                reduction="none",
            )
            loss = loss.detach().cpu()

            nrl_loss = loss[torch.where(labels != -100)]
            rel_loss = loss[torch.where(labels_part != -100)]
            nrl_loss = (torch.nansum(nrl_loss) - torch.nansum(rel_loss)) / (
                nrl_loss.size(0) - rel_loss.size(0)
            )
            rel_loss = torch.nanmean(rel_loss)

            return rel_loss, nrl_loss

        if batch_size is None:
            batch_size = self.valid_meta_batchsize

        self.meta_learner.eval()

        # Eval batch
        logs = defaultdict(list)

        for _ in range(batch_size):

            # Get data
            data_loader.eval()
            support_batch, query_batch = next(data_loader)

            (
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                supp_tgt_inputs_part,
                supp_labels_part,
            ) = support_batch
            (
                query_src_inputs,
                query_tgt_inputs,
                query_labels,
                query_tgt_inputs_part,
                query_labels_part,
            ) = query_batch

            # Clone the model for task-specific adaptation
            task_model = self.meta_learner.clone()

            # ==================================================================
            # Pre-adapt loss
            # ==================================================================

            # Get the losses pre-adaptation
            supp_pre_rel_loss, supp_pre_nrl_loss = get_rel_nrl_loss(
                task_model,
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                supp_labels_part,
            )

            query_pre_rel_loss, query_pre_nrl_loss = get_rel_nrl_loss(
                task_model,
                query_src_inputs,
                query_tgt_inputs,
                query_labels,
                query_labels_part,
            )

            logs[f"{split}/supp_pre_loss/rel"] += [supp_pre_rel_loss]
            logs[f"{split}/supp_pre_loss/nrl"] += [supp_pre_nrl_loss]

            logs[f"{split}/query_pre_loss/rel"] += [query_pre_rel_loss]
            logs[f"{split}/query_pre_loss/nrl"] += [query_pre_nrl_loss]

            # ==================================================================
            # Support / Inner loop optimization
            # ==================================================================

            task_model, _, _ = self.gmbl_step(
                task_model,
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                adapt_steps=self.valid_k,
                first_order=True,
            )

            # ==================================================================
            # Post-adapt loss
            # ==================================================================

            # Get the losses post-adaptation
            supp_post_rel_loss, supp_post_nrl_loss = get_rel_nrl_loss(
                task_model,
                supp_src_inputs,
                supp_tgt_inputs,
                supp_labels,
                supp_labels_part,
            )

            query_post_rel_loss, query_post_nrl_loss = get_rel_nrl_loss(
                task_model,
                query_src_inputs,
                query_tgt_inputs,
                query_labels,
                query_labels_part,
            )

            logs[f"{split}/supp_post_loss/rel"] += [supp_post_rel_loss]
            logs[f"{split}/supp_post_loss/nrl"] += [supp_post_nrl_loss]

            logs[f"{split}/query_post_loss/rel"] += [query_post_rel_loss]
            logs[f"{split}/query_post_loss/nrl"] += [query_post_nrl_loss]

            # ==================================================================
            # Adaptation improvement
            # ==================================================================

            # Compute the loss improvement post-adaptation
            logs[f"{split}/supp_imprv_mor/rel"] += [
                1 - supp_post_rel_loss / supp_pre_rel_loss
            ]
            logs[f"{split}/query_imprv_mor/rel"] += [
                1 - query_post_rel_loss / query_pre_rel_loss
            ]
            logs[f"{split}/cross_imprv_mor/rel"] += [
                1 - query_post_rel_loss / supp_pre_rel_loss
            ]
            logs[f"{split}/cross_imprv_rate_mor/rel"] += [
                (query_pre_rel_loss - query_post_rel_loss)
                / (supp_pre_rel_loss - supp_post_rel_loss)
            ]

            logs[f"{split}/supp_imprv_mor/nrl"] += [
                1 - supp_post_nrl_loss / supp_pre_nrl_loss
            ]
            logs[f"{split}/query_imprv_mor/nrl"] += [
                1 - query_post_nrl_loss / query_pre_nrl_loss
            ]
            logs[f"{split}/cross_imprv_mor/nrl"] += [
                1 - query_post_nrl_loss / supp_pre_nrl_loss
            ]
            logs[f"{split}/cross_imprv_mor/nrl"] += [
                1 - query_post_nrl_loss / supp_pre_nrl_loss
            ]
            logs[f"{split}/cross_imprv_rate_mor/nrl"] += [
                (query_pre_nrl_loss - query_post_nrl_loss)
                / (supp_pre_nrl_loss - supp_post_nrl_loss)
            ]

            del task_model

        logs = dict(logs)
        logs = {k: torch.mean(torch.stack(v)) for k, v in logs.items()}

        logs[f"{split}/supp_imprv_rom/rel"] = (
            1 - logs[f"{split}/supp_post_loss/rel"] / logs[f"{split}/supp_pre_loss/rel"]
        )
        logs[f"{split}/query_imprv_rom/rel"] = (
            1
            - logs[f"{split}/query_post_loss/rel"] / logs[f"{split}/query_pre_loss/rel"]
        )
        logs[f"{split}/cross_imprv_rom/rel"] = (
            1
            - logs[f"{split}/query_post_loss/rel"] / logs[f"{split}/supp_pre_loss/rel"]
        )
        logs[f"{split}/cross_imprv_rate_rom/rel"] = (
            logs[f"{split}/query_pre_loss/rel"] - logs[f"{split}/query_post_loss/rel"]
        ) / (logs[f"{split}/supp_pre_loss/rel"] - logs[f"{split}/supp_post_loss/rel"])

        logs[f"{split}/supp_imprv_rom/nrl"] = (
            1 - logs[f"{split}/supp_post_loss/nrl"] / logs[f"{split}/supp_pre_loss/nrl"]
        )
        logs[f"{split}/query_imprv_rom/nrl"] = (
            1
            - logs[f"{split}/query_post_loss/nrl"] / logs[f"{split}/query_pre_loss/nrl"]
        )
        logs[f"{split}/cross_imprv_rom/nrl"] = (
            1
            - logs[f"{split}/query_post_loss/nrl"] / logs[f"{split}/supp_pre_loss/nrl"]
        )
        logs[f"{split}/cross_imprv_rate_rom/nrl"] = (
            logs[f"{split}/query_pre_loss/nrl"] - logs[f"{split}/query_post_loss/nrl"]
        ) / (logs[f"{split}/supp_pre_loss/nrl"] - logs[f"{split}/supp_post_loss/nrl"])

        logs["global_step"] = self.global_step

        return logs
