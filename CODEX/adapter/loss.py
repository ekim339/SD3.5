"""Feature-distillation objective for channel-adapter training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def adapter_loss(prediction: torch.Tensor, target: torch.Tensor,
                 mse_weight: float = 1.0, cosine_weight: float = 0.1):
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    mse = F.mse_loss(prediction, target)
    cosine = (1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)).mean()
    total = float(mse_weight) * mse + float(cosine_weight) * cosine
    return {"total": total, "mse": mse, "cosine_distance": cosine}
