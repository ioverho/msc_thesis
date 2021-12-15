import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class char2word(nn.Module):
    """Character to word embeddings.

    """

    def __init__(
        self,
        vocab_len: int,
        embedding_dim: int = 256,
        h_dim: int = 256,
        bidirectional: bool = True,
        out_dim: int = 256,
        padding_idx: int = 1,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.h_dim = h_dim
        self.padding_idx = padding_idx

        self.embed = nn.Embedding(
            num_embeddings=vocab_len,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=h_dim,
            num_layers=1,
            bidirectional=bidirectional,
        )

        self.out_project = nn.Linear(
            in_features=(2 if bidirectional else 1) * h_dim, out_features=out_dim
        )

    def forward(self, chars: torch.Tensor, char_lens: torch.Tensor):

        c_embeds = self.embed(chars)

        packed_c_embeds = pack_padded_sequence(
            c_embeds, char_lens, enforce_sorted=False
        )

        _, h_T_out = self.rnn(packed_c_embeds)

        h_T_out = h_T_out.reshape(-1, (2 if self.bidirectional else 1) * self.h_dim)

        c2w_embeds = self.out_project(h_T_out)

        return c2w_embeds


class residual_lstm(nn.Module):
    """LSTM with (optional) residual connection.

    Also handles dropout internally, and can handle both packed/padded sequences.

    """
    def __init__(self, residual: bool = True, dropout_p: float = 0.0, **lstm_kwargs) -> None:
        super().__init__()

        self.residual = residual

        self.lstm = nn.LSTM(**lstm_kwargs)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):

        ht_out, _ = self.lstm(x)

        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            ht_out, lens = pad_packed_sequence(ht_out)
            ht_out = self.dropout(ht_out)
            if self.residual:
                x_, _ = pad_packed_sequence(x)
                ht_out = pack_padded_sequence(x_ + ht_out, lens, enforce_sorted=False)
            else:
                ht_out = pack_padded_sequence(ht_out, lens, enforce_sorted=False)
        else:
            ht_out = self.dropout(ht_out)
            if self.residual:
                ht_out = x + ht_out

        return ht_out


class residual_mlp_layer(nn.Module):
    """MLP layer with residual connection.

    """
    def __init__(self, **linear_kwargs) -> None:
        super().__init__()

        self.linear = nn.Linear(**linear_kwargs)

    def forward(self, x):

        h = torch.tanh(self.linear(x))
        h = h + x

        return h
