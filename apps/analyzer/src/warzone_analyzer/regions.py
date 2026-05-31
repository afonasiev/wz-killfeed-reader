from __future__ import annotations

import numpy as np

from .models import Region


def crop_region(frame: np.ndarray, region: Region) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = int(width * region.x)
    y1 = int(height * region.y)
    x2 = int(width * min(region.x + region.width, 1.0))
    y2 = int(height * min(region.y + region.height, 1.0))
    return frame[y1:y2, x1:x2]

