"""ConvLSTM baseline for spatiotemporal NDVI prediction.

Encoder-decoder architecture using ConvLSTM layers to process the full
140-timestep NDVI sequence and predict the target season.

Architecture:
    Input: (B, T, 1, H, W) where T=140 timesteps
    -> ConvLSTM encoder (2-3 layers, hidden dims [32, 64])
    -> Final hidden state -> Conv decoder -> (B, 1, H, W)

Total parameters: ~3-5M depending on configuration.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with convolutional gates.

    Implements the ConvLSTM formulation from Shi et al. (2015):
    gates use convolutions instead of fully-connected layers to
    preserve spatial structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        """Initialize ConvLSTM cell.

        Args:
            input_dim: Number of input channels.
            hidden_dim: Number of hidden channels.
            kernel_size: Convolution kernel size.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Combined gate convolution (input, forget, cell, output)
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one timestep.

        Args:
            x: Input tensor (B, C_in, H, W).
            state: Optional tuple of (h, c) each (B, hidden_dim, H, W).

        Returns:
            Tuple of (h_next, c_next).
        """
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMEncoder(nn.Module):
    """Multi-layer ConvLSTM encoder.

    Processes a sequence of spatial inputs and returns the final hidden state
    from the last layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
    ):
        """Initialize multi-layer ConvLSTM encoder.

        Args:
            input_dim: Number of input channels.
            hidden_dims: Hidden dimensions for each layer.
            kernel_size: Convolution kernel size for all layers.
        """
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims

        layers = []
        for i, h_dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layers.append(ConvLSTMCell(in_dim, h_dim, kernel_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process full sequence and return final hidden state.

        Args:
            x: Input sequence (B, T, C, H, W).

        Returns:
            Final hidden state from last layer (B, hidden_dims[-1], H, W).
        """
        B, T, C, H, W = x.shape

        # Initialize states for all layers
        states = [None] * self.num_layers

        for t in range(T):
            input_t = x[:, t]  # (B, C, H, W)
            for layer_idx, cell in enumerate(self.layers):
                h, c = cell(input_t, states[layer_idx])
                states[layer_idx] = (h, c)
                input_t = h  # Output of this layer is input to next

        # Return final hidden state of last layer
        return states[-1][0]  # (B, hidden_dims[-1], H, W)


class ConvDecoder(nn.Module):
    """Convolutional decoder to produce single-channel prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Initialize convolutional decoder.

        Args:
            input_dim: Input channels (from encoder hidden state).
            hidden_dim: Intermediate hidden channels.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden state to prediction.

        Args:
            h: Hidden state (B, C, H, W).

        Returns:
            Prediction (B, 1, H, W).
        """
        return self.decoder(h)


class ConvLSTMModel(nn.Module):
    """Full ConvLSTM encoder-decoder model for NDVI prediction.

    Takes the full 140-timestep NDVI sequence and predicts a single
    target timestep.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        kernel_size: int = 3,
        decoder_hidden: int = 64,
    ):
        """Initialize ConvLSTM model.

        Args:
            hidden_dims: Hidden dimensions per ConvLSTM layer.
                Default: [32, 64] (~3M params).
            kernel_size: Convolution kernel size.
            decoder_hidden: Hidden dim for decoder.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]

        self.encoder = ConvLSTMEncoder(
            input_dim=1, hidden_dims=hidden_dims, kernel_size=kernel_size,
        )
        self.decoder = ConvDecoder(
            input_dim=hidden_dims[-1], hidden_dim=decoder_hidden,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: NDVI sequence (B, T, 1, H, W) or (B, T, H, W).
                If 4D, a channel dimension is added.

        Returns:
            Prediction (B, 1, H, W).
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, T, H, W) -> (B, T, 1, H, W)

        h = self.encoder(x)
        return self.decoder(h)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
