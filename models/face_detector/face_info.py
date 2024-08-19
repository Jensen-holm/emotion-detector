from dataclasses import dataclass
import numpy as np


@dataclass
class FaceInfo:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        self.rows = np.sqrt(np.square(self.x1 - self.x2))
        self.cols = np.sqrt(np.square(self.y1 - self.y2))
