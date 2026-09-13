"""Supervised contrastive loss over a complete minibatch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """The standard label-based SupCon objective for one view per batch row.

    Every non-self row with the same label is a positive. All rows with a
    different label participate in the denominator as negatives.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "SupCon features must have shape [batch, dimension], got {}".format(
                    tuple(features.shape)
                )
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [batch]")
        if features.shape[0] < 2:
            raise ValueError("SupCon requires at least two samples")

        # Computing the similarity matrix in float32 avoids fp16 overflow while
        # retaining gradients back into an autocast glyph head.
        normalized = F.normalize(features.float(), p=2, dim=1)
        logits = normalized @ normalized.transpose(0, 1)
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = labels.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        positive_counts = positive_mask.sum(dim=1)
        if torch.any(positive_counts == 0):
            bad = torch.nonzero(positive_counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every SupCon anchor needs another sample with its label; "
                "anchors without positives: {}".format(bad)
            )

        log_denominator = torch.logsumexp(
            logits.masked_fill(self_mask, float("-inf")), dim=1
        )
        log_probability = logits - log_denominator[:, None]
        mean_positive_log_probability = (
            log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_counts
        )
        return -mean_positive_log_probability.mean()
