from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import grad

import learn2learn as l2l
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module


def clone_parameter_dict(param_dict):
    return {k: p.clone() for k, p in param_dict.items()}

class MarianGBML(BaseLearner):
    def __init__(self, model, lr, lrs, first_order):
        super().__init__()

        self.model = model

        self.lr = lr
        if lrs is None:
            self.lrs = nn.ParameterDict(
                {
                    name:
                        nn.Parameter(torch.tensor(lr, dtype=torch.float), requires_grad=True)
                    for name in self.updateable_params.keys()
                }
            )

        else:
            self.lrs = lrs

        self.first_order = first_order

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def updateable_params(self):
        raise NotImplementedError()

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
            push_to_device=False,
            **kwargs
        )

        logits = self.classify(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            features,
            decoder_input_ids,
            decoder_attention_mask,
        )

        return logits

    def adapt(self, loss, first_order=None):
        """Takes a gradient step on the loss and updates the cloned parameters in place.

        Args:
            loss (tensor): loss to minimize upon update
            first_order (bool, optional): whether to use first- or second-order updates. Defaults to self.first_order.
        """

        if first_order is None:
            first_order = self.first_order
        second_order = not first_order

        grads = grad(
            loss,
            list(self.updateable_params.values()),
            retain_graph=second_order,
            create_graph=second_order,
            allow_unused=False,
        )

        for i, (name, param) in enumerate(self.updateable_params.items()):
            param.update = - self.lrs[name] * grads[i]

        self.model = update_module(
            self.model,
            updates=None,
            )

    def clone(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return self.__class__(
            clone_module(self.model), lr=self.lr, lrs=clone_parameter_dict(self.lrs),
        )

class MarianDecMAML(MarianGBML):

    def __init__(self, model, lr, lrs=None, first_order: bool = False):
        super().__init__(model, lr, lrs, first_order)

    @property
    def updateable_params(self):

        params = OrderedDict(
                    [
                        (name.replace(".", "_"), param)
                        for name, param in self.model.model.named_parameters()
                        if "decoder.layers" in name] + [
                            ("lm_head", next(self.model.lm_head.parameters()))
                            ]
                        )

        return params

    def extract_features(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
    ):

        if push_to_device:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        encoder_output = self.model.model.encoder.forward(
            input_ids,
            attention_mask,
            **kwargs
            ).last_hidden_state

        return encoder_output

    def classify(
        self,
        input_ids,
        attention_mask,
        features,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
        ):

        if push_to_device:
            features = features.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            decoder_attention_mask = decoder_attention_mask.to(self.device)

        decoder_output = self.model.model.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=features,
            **kwargs
        )

        logits = self.model.lm_head(decoder_output.last_hidden_state) + self.model.final_logits_bias

        return logits

class MarianDecBOIL(MarianGBML):

    def __init__(self, model, lr, lrs=None, first_order: bool = False):
        super().__init__(model, lr, lrs, first_order)

    @property
    def updateable_params(self):

        params = OrderedDict(
                    [
                        (name.replace(".", "_"), param)
                        for name, param in self.model.model.named_parameters()
                        if "decoder.layers" in name
                        ]
                        )

        return params

    def extract_features(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
    ):

        if push_to_device:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        encoder_output = self.model.model.encoder.forward(
            input_ids,
            attention_mask,
            **kwargs
            ).last_hidden_state

        return encoder_output

    def classify(
        self,
        input_ids,
        attention_mask,
        features,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
        ):

        if push_to_device:
            features = features.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            decoder_attention_mask = decoder_attention_mask.to(self.device)

        decoder_output = self.model.model.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=features,
            **kwargs
        )

        logits = self.model.lm_head(decoder_output.last_hidden_state) + self.model.final_logits_bias

        return logits

class MarianDecANIL(MarianGBML):

    def __init__(self, model, lr, lrs=None, first_order: bool = False):
        super().__init__(model, lr, lrs, first_order)

    @property
    def updateable_params(self):

        params = OrderedDict(
            [
                ("lm_head", next(self.model.lm_head.parameters()))
                ]
            )

        return params

    def extract_features(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
    ):

        if push_to_device:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            decoder_attention_mask = decoder_attention_mask.to(self.device)

        features = self.model.model.forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            **kwargs
            ).last_hidden_state

        return features

    def classify(
        self,
        input_ids,
        attention_mask,
        features,
        decoder_input_ids,
        decoder_attention_mask,
        push_to_device: bool = False,
        **kwargs
        ):

        logits = self.model.lm_head(features) + self.model.final_logits_bias

        return logits
