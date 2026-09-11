"""
models.py

Phase 2 model architectures. Built on PyTorch's standard nn.LSTM/nn.GRU
layers rather than a from-scratch cell implementation - nobody hand-derives
LSTM gate equations for a project like this, and hand-rolling one would
just be a less-tested reimplementation of what the framework already
provides.

Each class here is one architecture from Section 5.6.1 of the proposal.
"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    """Section 5.6.1: the univariate floor for H1. Single unidirectional
    LSTM layer, trained on BTC log-return alone (input_size=1) rather
    than the full 19-feature vector.

    hidden_size isn't specified in the proposal for this baseline (only
    the Alpha Agent's 64-per-direction and the Beta Agent's 32 are
    pinned). 64 is chosen here to sit in the same capacity range as the
    Alpha Agent's Bi-LSTM stack, so the comparison isn't skewed by one
    model simply having far more capacity than the other.
    """

    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


class VanillaGRU(nn.Module):
    """Section 5.6.1: "Single GRU layer with matched parameter count" -
    matched to VanillaLSTM(hidden_size=64), which has 17,152 parameters.
    A GRU cell has fewer gates than an LSTM cell (2 vs 3), so it needs
    more hidden units to reach the same total parameter count rather
    than the same hidden_size. hidden_size=74 was chosen by direct
    parameter-count comparison: nn.GRU(input_size=1, hidden_size=74) has
    17,094 parameters, the closest integer match found (LSTM=17,152).
    """

    def __init__(self, input_size=1, hidden_size=74):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


class StandardBiLSTM(nn.Module):
    """Section 5.6.1: two-layer Bi-LSTM without attention, on the full
    19-feature vector - isolates what the Alpha Agent's self-attention
    layer specifically contributes (H2). Matches the Alpha Agent's own
    Bi-LSTM stack hyperparameters (Section 5.2): 2 stacked bidirectional
    layers, 64 hidden units per direction, dropout 0.2 between layers.

    Uses the final hidden state from each direction (h_n), not the
    output at the last time step (out[:, -1, :]). For a bidirectional
    LSTM those are NOT the same thing: the backward direction processes
    the sequence in reverse, so its output at the *last* time step has
    only just started (seen one step), while its entry in h_n is its
    state after genuinely processing the whole sequence backward. Using
    out[:, -1, :] would silently hand the backward half of the model
    almost no context.
    """

    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size). The last layer's
        # forward and backward final states are the last two entries.
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        combined = torch.cat([forward_final, backward_final], dim=-1)
        return self.head(combined).squeeze(-1)


class AlphaAgent(nn.Module):
    """Section 5.2: the real model. Same Bi-LSTM stack as
    StandardBiLSTM (2 layers, 64 units/direction, dropout 0.2) -
    deliberately identical, since H2 is specifically about isolating
    what self-attention adds on top of that stack, not about a
    different stack as well. The only architectural difference from
    StandardBiLSTM is what happens after the LSTM: a self-attention
    layer over the full 60-step output sequence, instead of just using
    the stack's own final hidden state.

    Scaled dot-product self-attention (Vaswani et al.), single head, via
    nn.MultiheadAttention. Query = the Bi-LSTM's output at the LAST time
    step only (today - the vantage point the forecast is made from).
    Keys/values = the full 60-step output sequence. This lets today's
    forecast pull in whichever of the 60 days is most relevant, rather
    than only what survived being compressed into the recurrent state
    by the time processing reaches today - exactly the "instead of
    relying solely on the final hidden state" framing in Section 5.2.
    """

    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.2, num_heads=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        attn_dim = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True,
        )
        self.head = nn.Linear(attn_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)              # (batch, 60, hidden*2)
        query = out[:, -1:, :]             # (batch, 1, hidden*2) - "today"
        attn_out, _ = self.attention(query, out, out)
        summary = attn_out.squeeze(1)      # (batch, hidden*2)
        return self.head(summary).squeeze(-1)
