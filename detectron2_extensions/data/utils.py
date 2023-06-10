import torch
from typing import Sequence, Union
import torch.nn.functional as F


def extract_window_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.

    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    image = image.unsqueeze(0).float()
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    x1 = max(x1 - p, 0)
    y1 = max(y1 - p, 0)
    x2 = min(x2 + p, W)
    y2 = min(y2 + p, H)
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # Zero pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = F.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = F.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    ).squeeze(0)

    return window
