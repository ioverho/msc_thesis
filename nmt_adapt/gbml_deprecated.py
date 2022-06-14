import torch
import torch.nn as nn
from torch.autograd import grad

import learn2learn as l2l
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module


def clone_parameter_dict(param_dict):
    return {k: p.clone() for k, p in param_dict.items()}


class MarianGBML(BaseLearner):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def extract_features(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):

        device = self.device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)

        return self.feature_extractor.forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            **kwargs
        ).last_hidden_state

    def classify(self, features):

        device = self.device

        features = features.to(device)

        logits = self.classifier.forward(features) + self.model.final_logits_bias

        return logits

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):

        features = self.extract_features(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            **kwargs
        )

        logits = self.classify(features)

        return logits


class MarianMAMLpp(MarianGBML):
    """ High-level implementation of 'MAML++'

    Args
        model (Module) - Module to be wrapped.
        lr (float) - Fast adaptation learning rate.
        lslr (bool) - Whether to use Per-Layer Learning Rates and Gradient Directions (LSLR) or not.
        lrs (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list as learning rates for fast-adaptation.
        first_order (bool, *optional*, default=False) - Whether to use the first-order approximation of MAML. (FOMAML)
        allow_unused (bool, *optional*, default=None) - Whether to allow differentiation of unused parameters. Defaults to `allow_nograd`.
        allow_nograd (bool, *optional*, default=False) - Whether to allow adaptation with parameters that have `requires_grad = False`.
    """

    def __init__(
        self,
        model,
        lr,
        lrs=None,
        non_adaptable_layers=None,
        first_order=False,
        allow_unused=None,
        allow_nograd=False,
    ):
        super().__init__()

        self.nmt_model = model
        self.lr = lr
        self.non_adaptable_layers = non_adaptable_layers

        self.lrs = self._init_lslr_parameters() if lrs is None else lrs

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def _init_lslr_parameters(self):

        default_lr = torch.tensor(self.lr)

        lslrs = dict()
        for name, _ in self.nmt_model.model.named_parameters():
            if name not in self.non_adaptable_layers:
                lslrs["_".join(name.split(".")[:3])] = nn.Parameter(
                    torch.tensor([default_lr], dtype=torch.float), requires_grad=True
                )

        lslrs["lm_head"] = nn.Parameter(
            torch.tensor([default_lr], dtype=torch.float), requires_grad=True
        )

        return nn.ParameterDict(lslrs)

    def adapt(self, loss, first_order=None):
        """
        **Description**
        Takes a gradient step on the loss and updates the cloned parameters in place.
        **Arguments**
        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order

        body_params = [p for p in self.nmt_model.model.parameters() if p.requires_grad]
        head_params = [
            p for p in self.nmt_model.lm_head.parameters() if p.requires_grad
        ]

        grads = grad(
            loss,
            body_params + head_params,
            retain_graph=second_order,
            create_graph=second_order,
            allow_unused=False,
        )

        i = 0
        for n, p in self.nmt_model.model.named_parameters():
            if p.requires_grad:
                if n not in self.non_adaptable_layers:
                    l = "_".join(n.split(".")[:3])
                    p.update = -self.lrs[l] * grads[i]

                i += 1

        for p in self.nmt_model.lm_head.parameters():
            if p.requires_grad and n not in self.non_adaptable_layers:
                p.update = -self.lrs["lm_head"] * grads[i]

            i += 1

        self.nmt_model = update_module(self.nmt_model)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**
        Returns a `MAMLpp`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.
        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().
        **Arguments**
        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MarianMAMLpp(
            clone_module(self.nmt_model),
            lr=self.lr,
            lrs=clone_parameter_dict(self.lrs),
            non_adaptable_layers=self.non_adaptable_layers,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )


class MarianAnil(MarianGBML):
    """_summary_

    Args:
        model (_type_): _description_
        lr (_type_): _description_
        lrs (_type_, optional): _description_. Defaults to None.
        first_order (bool, optional): _description_. Defaults to False.

    """

    def __init__(self, model, lr, lrs=None, first_order: bool = False):
        super().__init__()

        self.model = model

        self.feature_extractor = self.model.model

        self.classifier = self.model.lm_head

        self.lr = lr
        if lrs is None:
            self.lrs = nn.ParameterDict(
                {
                    "lm_head": nn.Parameter(
                        torch.tensor(lr, dtype=torch.float), requires_grad=True
                    )
                }
            )
        else:
            self.lrs = lrs

        self.first_order = first_order

    def adapt(self, loss, first_order=None):
        """Takes a gradient step on the loss and updates the cloned parameters in place.

        Args:
            loss (tensor): loss to minimize upon update
            first_order (bool, optional): whether to use first- or second-order updates. Defaults to self.first_order.
        """

        if first_order is None:
            first_order = self.first_order
        second_order = not first_order

        head_params = [p for p in self.classifier.parameters()]

        grads = grad(
            loss,
            head_params,
            retain_graph=second_order,
            create_graph=second_order,
            allow_unused=False,
        )

        for i, p in enumerate(head_params):
            p.update = -self.lrs["lm_head"] * grads[i]

        self.classifier = update_module(self.classifier)

    def clone(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return MarianAnil(
            clone_module(self.model), lr=self.lr, lrs=clone_parameter_dict(self.lrs),
        )

