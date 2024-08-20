from types import NoneType
import cv2
import os

from cv2.typing import MatLike
from .face_info import FaceInfo


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detection_yunet_2023mar.onnx",
)


class FaceDetector:
    __slots__ = ["__model"]
    __MEAN_SUB_VALS = (104.0, 117.0, 123.0)
    __BLOB_SIZE = (320, 320)

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.__model = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=self.__BLOB_SIZE,
            score_threshold=confidence_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=0,
            target_id=0,
        )

    def predict(self, frame_input: MatLike) -> list[FaceInfo]:
        """returns a list of bounding box information for each face detected by yunet"""
        resized_frame = cv2.resize(frame_input, self.__BLOB_SIZE)
        faces = self.__model.detect(resized_frame)
        if isinstance(faces[1], NoneType):
            return []

        face_infos: list[FaceInfo] = []
        input_height, input_width, _ = frame_input.shape

        for face_info in faces[1]:
            x1, y1, w, h = face_info[:4]  # ignoring face landmarks
            width_scaled = int(w * (input_width / self.__BLOB_SIZE[0]))
            height_scaled = int(h * (input_height / self.__BLOB_SIZE[1]))
            x1 = int(x1 * (input_width / self.__BLOB_SIZE[0]))
            y1 = int(y1 * (input_height / self.__BLOB_SIZE[1]))
            y2 = y1 + height_scaled
            x2 = x1 + width_scaled
            coords = (x1, y1, x2, y2)
            if any([coord < 0 for coord in coords]):
                continue
            face_infos.append(FaceInfo(*coords))

        return face_infos
