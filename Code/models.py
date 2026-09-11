"""
models.py

Phase 2 model architectures. Built on PyTorch's standard nn.LSTM/nn.GRU
layers rather than a from-scratch cell implementation - nobody hand-derives
LSTM gate equations for a project like this, and hand-rolling one would
just be a less-tested reimplementation of what the framework already
provides.

Each class here is one architecture from Section 5.6.1 of the proposal.
"""

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
