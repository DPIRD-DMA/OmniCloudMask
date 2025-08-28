from typing import Optional

import torch
import torch.nn.functional as F
from fastai.metrics import DiceMulti
from torch import Tensor


class DiceMultiStrip(DiceMulti):
    """DiceMulti that only looks at the first element if y is a tuple"""

    def accumulate(self, learn):
        # Temporarily modify learn.yb (the underlying batch) instead of learn.y
        yb_backup = learn.yb

        # Extract targets if it's a tuple
        if isinstance(learn.yb, (tuple, list)) and len(learn.yb) > 0:
            if isinstance(learn.yb[0], (tuple, list)):
                # If yb contains tuples, extract the first element of each tuple
                learn.yb = (learn.yb[0][0],)  # Just the targets, keep it as a tuple
            else:
                learn.yb = (learn.yb[0],)  # Wrap in tuple to maintain structure

        try:
            # Call the parent accumulate method
            super().accumulate(learn)
        finally:
            # Always restore the original yb
            learn.yb = yb_backup


class CrossEntropyLossFlatImageTypeWeighted:
    def __init__(
        self,
        axis: int = 1,
        ignore_index: int = 99,
        class_weights: Optional[Tensor] = None,
    ):
        self.axis = axis
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def __call__(self, preds: Tensor, targets: Tensor, image_weights: Tensor) -> Tensor:
        # preds: [bs, classes, h, w]
        # targets: [bs, h, w]
        # image_weights:[bs]

        weights = image_weights.to(targets.device, dtype=torch.float)

        _, classes, _, _ = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).contiguous().view(-1, classes)

        targets_flat = targets.reshape(-1)

        # Get CEL with mask value, no reduction, so we can apply image weights later
        pixel_losses = F.cross_entropy(
            preds_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction="none",
            weight=self.class_weights,
        )
        # Some images have large areas of the ignore index,
        # So we take the mean of each image loss
        image_losses = pixel_losses.reshape(targets.shape[0], -1).mean(dim=1)

        # Apply the weights
        weighted_losses = image_losses * weights

        # Get mean and return
        return weighted_losses.mean()
