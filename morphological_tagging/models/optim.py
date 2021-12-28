from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvSqrtWithLinearWarmupScheduler():
    """A learning rate scheduler according to the Attention is All You Need paper.
    Unlike original implementation, maximum lr scale (immediately after warmup) is 1.
    The actual learning rates are controlled by setting the defaults.

    Args:
        optimizer (torch.optim.Optimizer): optimizer.
        default_lrs (List[Dict]): a list of dicts, with each dict being a parameter-group for the optimizer initialization.
        n_warmup_steps (int): number of warmup steps to take before reaching max learning rate.

    Inspired by:
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
    """

    def __init__(self, optimizer, default_lrs, n_warmup_steps):
        self._optimizer = optimizer
        self._default_lrs = deepcopy([pg['lr'] for pg in default_lrs])
        self.n_warmup_steps = n_warmup_steps

        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def step(self):
        """Step with the inner optimizer"""
        self.step_and_update_lr()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):

        lr_scale = (self.n_warmup_steps ** (0.5)) * min(
            self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5)
        )

        return lr_scale

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1

        for p_group, default_lr in zip(
            self._optimizer.param_groups, self._default_lrs
        ):
            p_group["lr"] = default_lr * self._get_lr_scale()
