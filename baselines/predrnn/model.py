"""PredRNN baseline for spatiotemporal NDVI prediction.

Implements PredRNN (Wang et al., 2017) with Spatiotemporal LSTM (ST-LSTM) cells
that feature a zigzag memory flow between layers. The key innovation is a
spatiotemporal memory cell (M) that flows across layers within each timestep,
capturing spatial correlations that the standard temporal memory (C) misses.

Architecture:
    Input: (B, T, 1, H, W) where T=140 timesteps
    -> ST-LSTM stack with zigzag memory flow
    -> Final hidden state -> Conv decoder -> (B, 1, H, W)

Total parameters: ~5-10M depending on configuration.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class SpatiotemporalLSTMCell(nn.Module):
    """Spatiotemporal LSTM cell from PredRNN.

    Extends ConvLSTM with an additional spatiotemporal memory (M) that
    flows across layers (zigzag pattern) while the temporal memory (C)
    flows across timesteps within each layer.

    The gate equations:
        g_t = tanh(W_xg * x + W_hg * h_{t-1})
        i_t = sigmoid(W_xi * x + W_hi * h_{t-1})
        f_t = sigmoid(W_xf * x + W_hf * h_{t-1})
        C_t = f_t * C_{t-1} + i_t * g_t

        g'_t = tanh(W_xg' * x + W_mg' * M_{l-1})
        i'_t = sigmoid(W_xi' * x + W_mi' * M_{l-1})
        f'_t = sigmoid(W_xf' * x + W_mf' * M_{l-1})
        M_t = f'_t * M_{l-1} + i'_t * g'_t

        o_t = sigmoid(W_xo * x + W_ho * h_{t-1} + W_co * C_t + W_mo * M_t)
        h_t = o_t * tanh(W_1x1 * [C_t, M_t])
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        """Initialize ST-LSTM cell.

        Args:
            input_dim: Number of input channels.
            hidden_dim: Number of hidden channels (same for C and M).
            kernel_size: Convolution kernel size.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Temporal gates: input, forget, cell, output (standard LSTM part)
        self.conv_temporal = nn.Conv2d(
            input_dim + hidden_dim, 3 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

        # Spatiotemporal gates: input', forget', cell' (for M memory)
        self.conv_spatial = nn.Conv2d(
            input_dim + hidden_dim, 3 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

        # Output gate combines both memories
        self.conv_output = nn.Conv2d(
            input_dim + hidden_dim + 2 * hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

        # 1x1 conv to combine C and M for final hidden state
        self.conv_combine = nn.Conv2d(
            2 * hidden_dim, hidden_dim, kernel_size=1, bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for one timestep.

        Args:
            x: Input (B, C_in, H, W).
            h_prev: Previous hidden state (B, hidden_dim, H, W).
            c_prev: Previous temporal memory (B, hidden_dim, H, W).
            m_prev: Previous spatiotemporal memory from layer below (B, hidden_dim, H, W).

        Returns:
            Tuple of (h_next, c_next, m_next).
        """
        # Temporal gates (standard LSTM part)
        combined_h = torch.cat([x, h_prev], dim=1)
        temporal_gates = self.conv_temporal(combined_h)
        i, f, g = temporal_gates.chunk(3, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g

        # Spatiotemporal gates (M memory part)
        combined_m = torch.cat([x, m_prev], dim=1)
        spatial_gates = self.conv_spatial(combined_m)
        i_prime, f_prime, g_prime = spatial_gates.chunk(3, dim=1)
        i_prime = torch.sigmoid(i_prime)
        f_prime = torch.sigmoid(f_prime)
        g_prime = torch.tanh(g_prime)
        m_next = f_prime * m_prev + i_prime * g_prime

        # Output gate
        combined_all = torch.cat([x, h_prev, c_next, m_next], dim=1)
        o = torch.sigmoid(self.conv_output(combined_all))

        # Hidden state combines both memories
        h_next = o * torch.tanh(self.conv_combine(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next


class PredRNNModel(nn.Module):
    """PredRNN encoder-decoder model for NDVI prediction.

    Uses ST-LSTM layers with zigzag memory flow: the spatiotemporal memory M
    flows from layer 1 -> 2 -> ... -> L at each timestep, then from layer
    L -> L-1 -> ... -> 1 at the next timestep (zigzag pattern).
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        kernel_size: int = 3,
        decoder_hidden: int = 64,
    ):
        """Initialize PredRNN model.

        Args:
            hidden_dims: Hidden dimensions per ST-LSTM layer.
                Default: [64, 64, 64] (~8M params).
            kernel_size: Convolution kernel size.
            decoder_hidden: Hidden dim for decoder.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]

        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims

        # ST-LSTM layers
        layers = []
        for i, h_dim in enumerate(hidden_dims):
            in_dim = 1 if i == 0 else hidden_dims[i - 1]
            layers.append(SpatiotemporalLSTMCell(in_dim, h_dim, kernel_size))
        self.layers = nn.ModuleList(layers)

        # Conv decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], decoder_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_hidden, decoder_hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_hidden // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with zigzag M memory flow.

        M flows bottom-up through layers at each timestep (layer 1 -> L).
        At the next timestep, M from layer L feeds back into layer 1,
        creating a zigzag pattern across time and layers.

        Args:
            x: NDVI sequence (B, T, 1, H, W) or (B, T, H, W).

        Returns:
            Prediction (B, 1, H, W).
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, T, C, H, W = x.shape
        h, c, m = self._init_states(B, H, W, x.device, x.dtype)

        for t in range(T):
            h, c, m = self._process_timestep(x[:, t], h, c, m)

        return self.decoder(h[-1])

    def _init_states(
        self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype,
    ) -> Tuple[list, list, torch.Tensor]:
        """Initialize hidden states, cell states, and M memory."""
        h = [torch.zeros(B, d, H, W, device=device, dtype=dtype) for d in self.hidden_dims]
        c = [torch.zeros(B, d, H, W, device=device, dtype=dtype) for d in self.hidden_dims]
        m = torch.zeros(B, self.hidden_dims[0], H, W, device=device, dtype=dtype)
        return h, c, m

    def _process_timestep(
        self,
        input_t: torch.Tensor,
        h: list,
        c: list,
        m: torch.Tensor,
    ) -> Tuple[list, list, torch.Tensor]:
        """Process one timestep through all ST-LSTM layers (bottom-up).

        M flows from layer 0 -> 1 -> ... -> L. The final M from layer L
        will be used as the initial M for layer 0 at the next timestep.
        """
        m_current = m
        for layer_idx, cell in enumerate(self.layers):
            cell_input = input_t if layer_idx == 0 else h[layer_idx - 1]
            h[layer_idx], c[layer_idx], m_current = cell(
                cell_input, h[layer_idx], c[layer_idx], m_current,
            )
        return h, c, m_current

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
