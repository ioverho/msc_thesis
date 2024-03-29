from copy import deepcopy

class DummyScheduler:
    def __init__(self):
        pass

    def step(self):
        return None

    def lambda_step(self):
        pass

class LinearDecay:
    """A learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): optimizer.
        default_lrs (List[Dict]): a list of dicts, with each dict being a parameter-group for the optimizer initialization.
        n_warmup_steps (int): number of warmup steps to take before reaching max learning rate.

    """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self._default_lrs = deepcopy([pg["lr"] for pg in self.optimizer.param_groups])
        self.n_warmup_steps = n_warmup_steps

        self.n_steps = 0
        self._frozen = False

    def step(self):
        """Step the scheduler"""
        self._update_learning_rate()

    def _get_lr_scale(self):

        lr_scale = max(0.0, 1- float(self.n_steps) / float(self.n_warmup_steps))

        return lr_scale

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1

        for p_group, default_lr in zip(self.optimizer.param_groups, self._default_lrs):
            p_group["lr"] = default_lr * self._get_lr_scale()

    def lambda_step(self, fn):

        self._default_lrs = [
            fn(default_lr)
            for default_lr in self._default_lrs
            ]


class InvSqrtWithLinearWarmupScheduler:
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

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self._default_lrs = deepcopy([pg["lr"] for pg in self.optimizer.param_groups])
        self.n_warmup_steps = n_warmup_steps

        self.n_steps = 0
        self._frozen = False

    def step(self):
        """Step the scheduler"""
        self._update_learning_rate()

    def _get_lr_scale(self):

        lr_scale = (self.n_warmup_steps ** (0.5)) * min(
            self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5)
        )

        return lr_scale

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1

        for p_group, default_lr in zip(self.optimizer.param_groups, self._default_lrs):
            p_group["lr"] = default_lr * self._get_lr_scale()

    def lambda_step(self, fn):

        self._default_lrs = [
            fn(default_lr)
            for default_lr in self._default_lrs
            ]
