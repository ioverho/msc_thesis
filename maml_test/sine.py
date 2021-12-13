import numpy as np

import torch
import torch.distributions as D


class SampleSinuisoid:
    """Class for sampling sinusoids.

    Should emulate behaviour of initial MAML paper.

    Args:
        device (torch.device): deivce on which to create data. Defaults to 'cpu'.
        x_range (tuple, optional): Range of support. Sampled from continuous uniform distribution.
            Defaults to (-5.0, 5.0).
        amplitude_range (tuple, optional): Range of possible amplitudes for sampled sinusoids.
            Defaults to (0.1, 5.0).
        phase_range (tuple, optional): Range of possible phases for sampled sinusoids.
            Defaults to (0, np.pi).
        amp_params (tuple, optional): Beta distribution parameters. Defaults to (1, 1).
        phase_params (tuple, optional): Beta distribution parameters. Defaults to (1, 1).
    """

    def __init__(self,
                 device=torch.device('cpu'),
                 x_range=(-5.0, 5.0),
                 amplitude_range=(0.1, 5.0),
                 phase_range=(0, np.pi),
                 amp_params=(1, 1),
                 phase_params=(1, 1)
                 ):
        super().__init__()

        self.device = device

        self.x_range = x_range
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.amp_params = amp_params
        self.phase_params = phase_params

        self.support = D.Uniform(*self._tuple_to_tensor(self.x_range))
        self.amplitude = D.Beta(*self._tuple_to_tensor(self.amp_params))
        self.phase = D.Beta(*self._tuple_to_tensor(self.phase_params))

    def _tuple_to_tensor(self, t):

        tensors = []
        for val in t:
            tensors.append(torch.tensor([val],
                                        dtype=torch.float,
                                        device=self.device))

        return tensors

    def _min_max_scale(self, x, min, max):

        x_scaled = x * (max - min) + min

        return x_scaled

    def _sample_params(self, n):

        amps, phases = self.amplitude.sample((n,)), self.phase.sample((n,))

        amps, phases = self._scale_params(amps, phases)

        return amps, phases

    def _scale_params(self, amps, phases):

        amps = self._min_max_scale(amps,
                                   self.amplitude_range[0], self.amplitude_range[1])

        phases = self._min_max_scale(phases,
                                     self.phase_range[0], self.phase_range[1])

        return amps, phases

    def sample_task(self):

        self.cur_amplitude, self.cur_phase = self._sample_params(1)

    def sample_batch(self, batch_size):

        x = self.support.sample((batch_size, 1))
        y = self.transform(x)

        return x, y

    def transform(self, x, amp=None, phase=None):
        if amp == None:
            amp = self.cur_amplitude

        if phase == None:
            phase = self.cur_phase

        return amp * torch.sin(x + phase)

    def __str__(self) -> str:
        return f'Random Sinusoid. Amp: {self.cur_amplitude.item():2f}, '\
            + f'Phase: {self.cur_phase.item():2f}. Device: {self.device}'
