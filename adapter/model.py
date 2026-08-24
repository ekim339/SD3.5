"""Channel projection from residual style maps to TextCtrl style grids."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ChannelAdapter(nn.Module):
    """Pointwise projection `[B,256,16,16] -> [B,768,16,16]`."""

    def __init__(self, input_channels: int = 256, output_channels: int = 768) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.projection = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        expected = (self.input_channels, 16, 16)
        if residual.ndim != 4 or tuple(residual.shape[1:]) != expected:
            raise ValueError(
                f"Expected residual shape [B,{expected[0]},16,16], got {tuple(residual.shape)}"
            )
        output = self.projection(residual)
        if tuple(output.shape[1:]) != (self.output_channels, 16, 16):
            raise RuntimeError(f"Unexpected adapter output shape: {tuple(output.shape)}")
        return output


def load_adapter(checkpoint: str | Path, device: str | torch.device = "cpu") -> ChannelAdapter:
    """Load an adapter checkpoint written by `adapter.train`."""
    path = Path(checkpoint).expanduser()
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = payload.get("model_config", {})
    model = ChannelAdapter(**model_config)
    state = payload.get("adapter", payload)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
