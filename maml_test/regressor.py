import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SineRegressor(nn.Module):
    """Sinusoid regressor as suggested by Finn et al.
    """

    def __init__(self, hidden_dim: int = 40) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x) -> torch.Tensor:

        return self.net(x)

    def train(self):
        """Set model to train mode. Activates regularization.
        """

        self.net.train()

    def eval(self):
        """Set model to eval mode. Kills regularization.
        """

        self.net.eval()

    @property
    def device(self):
        """
        Hacky method for checking model device.
        Requires all parameters to be on same device.
        """
        return next(self.net.parameters()).device

    def freeze(self):
        """Freeze all module parameters. Will not train.
        """

        for param in self.net.parameters():
            param.requires_grad = False

    def thaw(self):
        """Unfreeze/defrost/thaw all module parameters. Will train.
        """

        for param in self.net.parameters():
            param.requires_grad = True

class BatchRegressor(nn.Module):

    def __init__(self, hidden_dim) -> None:
        super().__init__()

        self.encode_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.params_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

    def _get_params(self, x, dataset):
        h = self.encode_net(x)

        h = torch.max(h, dim=0)[0].unsqueeze(0)

        p = self.params_net(h)

        amp_hat, phase_hat = p[:, 0], p[:, 1]

        amp_hat, phase_hat = dataset._scale_params(amp_hat, phase_hat)

        return amp_hat, phase_hat

    def forward(self, x, dataset):

        amp_hat, phase_hat = self._get_params(x, dataset)

        y_hat = dataset.transform(x, amp=amp_hat, phase=phase_hat)

        return y_hat, amp_hat, phase_hat
