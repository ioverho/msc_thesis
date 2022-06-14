import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence

class SequenceMask(nn.Module):
    """Masks a sequence tensor randomly.
    A form of regularization/data augmentation.
    Expects the tensor to be of form [B,T] or [T,B] (i.e. no dimensions) and of type torch.long.

    Args:
        mask_p (float): the probability of a single entry in the sequence gets masked. Defaults to 0.
        mask_idx (int): the mask index replacing the given values
        ign_idx (int): an index to be ignored, for example, padding
    """

    def __init__(self, mask_p: float = 0.0, mask_idx: int = 0) -> None:
        super().__init__()

        self.mask_p = float(mask_p)
        self.register_buffer("mask_idx", torch.tensor(mask_idx, dtype=torch.long))

    def forward(self, x: torch.Tensor, ign_mask: torch.Tensor = None):

        if ign_mask is None:
            ign_mask = torch.ones_like(x)

        if self.training and self.mask_p > 0.0:
            x = torch.where(
                torch.logical_or(
                    torch.bernoulli(x, 1-self.mask_p), ign_mask == 0
                ),
                x,
                self.mask_idx,
            )

        return x

class LayerAttention(nn.Module):
    """A layer attention module.

    Args:
        L (int): number of layers to attend over
        u (float, optional): range for initiliazation. Defaults to 0.2.
        dropout (float, optional): probability of dropout. Defaults to 0.0.
    """

    def __init__(self, L: int, u: float = 0.2, dropout: float = 0.0) -> None:
        super().__init__()

        self.L = L
        self.u = u
        self.dropout = dropout

        self.h_w = nn.Parameter(torch.empty(self.L), requires_grad=True)
        self.c = nn.Parameter(torch.ones(1), requires_grad=True)
        init.uniform_(self.h_w, a=-self.u, b=self.u)

        self.layer_dropout = nn.Dropout(self.dropout)

    def forward(self, h: torch.Tensor):
        """Attends on L layers of h.

        Args:
            h (torch.Tensor): takes a torch tensor representing the hidden layers
                of a transformer. Assumed shape of [B, T, L, D] or [T, B, L, D].
        """

        h = h[:, :, -self.L:, :]

        alpha = torch.softmax(self.h_w, dim=0)

        alpha = self.layer_dropout(alpha)

        h_out = self.c * torch.sum((alpha.view(1, 1, -1, 1) * h), dim=2)

        return h_out

class TokenClassifier(nn.Module):

    def __init__(self, in_features, hidden_dim, out_features, L, dropout: float = 0.0, layer_dropout: float = 0.0, **unused_kwargs):
        super().__init__()

        self.layer_attn = LayerAttention(
            L=L,
            dropout=layer_dropout,
        )

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=in_features,
                out_features=hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=out_features,
            ),
        )

    def forward(self, features, tgt_input_ids, tokenizer):

        bpe_reps = self.layer_attn(features)

        batch_bpes_rep = []
        for bpe_seq, bpe_rep_seq in zip(tgt_input_ids, bpe_reps):

            sent_bpes_rep = []
            for bpe, bpe_rep in zip(tokenizer.convert_ids_to_tokens(bpe_seq), bpe_rep_seq):
                if bpe[0] == "▁":
                    sent_bpes_rep.append([bpe_rep])

                elif bpe not in set(tokenizer.all_special_tokens) - {tokenizer.unk_token}:
                    sent_bpes_rep[-1].append(bpe_rep)

            batch_bpes_rep.append(
                torch.stack(
                    [torch.mean(torch.stack(token_bpe_rep, dim=0), dim=0)
                    for token_bpe_rep in sent_bpes_rep
                    ],
                    dim=0)
            )

        tokens_rep = pad_sequence(
            batch_bpes_rep,
            batch_first=True,
            padding_value=0.0,
            )

        token_logits = self.clf(tokens_rep)

        return token_logits
