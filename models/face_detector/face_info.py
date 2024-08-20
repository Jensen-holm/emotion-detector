from dataclasses import dataclass
from cv2.typing import MatLike
from typing import Tuple
import numpy as np
import cv2


@dataclass
class FaceInfo:
    __slots__ = ["x1", "y1", "x2", "y2", "pt1", "pt2", "rows", "cols"]
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        self.pt1 = (self.x1, self.y1)
        self.pt2 = (self.x2, self.y2)

    def crop(self, frame: MatLike) -> MatLike:
        """crop the frame to jsut the pixels of this face"""
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    def overlay(self, frame: MatLike, overlay_img: MatLike) -> MatLike:
        """replace the face with an image of your choosing"""
        cropped_face = self.crop(frame).astype(np.float32)
        resized_overlay = cv2.resize(
            overlay_img,
            (cropped_face.shape[1], cropped_face.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        face_px_to_add = np.where(resized_overlay == 0, cropped_face, 0)
        add_result = cv2.add(resized_overlay, face_px_to_add)

        add_result = np.clip(add_result, 0, 255).astype(np.uint8)
        frame[self.y1 : self.y2, self.x1 : self.x2] = add_result
        return frame

    def draw_box(
        self, frame: MatLike, color: Tuple[int, int, int] = (0, 255, 0)
    ) -> MatLike:
        """draws a bounding box around the face"""
        cv2.rectangle(frame, self.pt1, self.pt2, color, thickness=2)
        return frame
