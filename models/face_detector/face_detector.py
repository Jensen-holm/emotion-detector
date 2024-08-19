import cv2
import os

from cv2.typing import MatLike
from .face_info import FaceInfo


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    # "version-slim-320_simplified.onnx",
    "yunet.onnx",
)


class FaceDetector:
    BLOB_SIZE = (320, 240)
    MEAN_SUB_VALUES = (127.0, 127.0, 127.0)

    def __init__(self) -> None:
        self.model = cv2.dnn.readNetFromONNX(MODEL_PATH)

    def _pre_process_input(self, frame_input: MatLike) -> MatLike:
        """
        converts the frame into a blob that the face detector model can understand.
        reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/tf/det_image.py
        """
        return cv2.dnn.blobFromImage(
            frame_input,
            1.0 / 128.0,
            self.BLOB_SIZE,
            self.MEAN_SUB_VALUES,
            swapRB=True,
        )

    def predict(self, frame_input: MatLike):
        """result in results =[background, face, x1, y1, x2, y2"""
        orig_w, orig_h, _ = frame_input.shape
        processed_frame = self._pre_process_input(frame_input)
        self.model.setInput(processed_frame)
        output = self.model.forward()
        return output
