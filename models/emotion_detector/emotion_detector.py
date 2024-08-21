from cv2.typing import MatLike
import numpy as np
import cv2
import os

from .emotion_map import load_emoji_map

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "emotion_detector.pb",
)


class EmotionDetector:
    __slots__ = ["emoji_map", "__model"]
    BLOB_SIZE = (48, 48)

    def __init__(self) -> None:
        """loads the emotion detector model into opencv.dnn.Net"""
        self.emoji_map = load_emoji_map()
        self.__model = cv2.dnn.readNetFromTensorflow(MODEL_PATH)

    def _pre_process_input(self, face_input: MatLike) -> MatLike:
        """
        recieves a cropped frame of a persons face, returns the blob that can go into the model.
        The model was trained on 48x48 pixel images of faces, so we need to compress the input image
        here into those specifications in order to get an accurate prediction.
        """
        gray_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)
        return cv2.dnn.blobFromImage(
            gray_input,
            1.0 / 255.0,
            self.BLOB_SIZE,
            (0, 0, 0),
            swapRB=False,
        )

    def predict(self, face_input: MatLike) -> MatLike:
        """
        detects emotion of the cropped face input image and returns the coresponding emoji.
        raises an exception if the detected emotion does not exits.
        """
        pre_processed_face = self._pre_process_input(face_input)
        self.__model.setInput(pre_processed_face)
        output = self.__model.forward()
        emotion = int(np.argmax(output))

        if emotion == 1:  # disgust = sad
            emotion = 4
        if emotion == 2:  # fear = anger
            emotion = 0

        if (emoji := self.emoji_map.get(emotion, None)) is None:
            raise Exception(
                f"no such emoji in the emoji map at emotion idx = {emotion}"
            )
        return emoji
