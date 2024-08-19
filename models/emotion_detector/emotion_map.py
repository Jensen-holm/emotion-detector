from cv2.typing import MatLike
import cv2


def load_emoji_map() -> dict[int, MatLike]:
    """
    loads a dictionary of the emoji's in ./assets that correspond to
    the emotion detection models output.
    details: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv
    """
    return {
        0: cv2.imread(""),
        1: cv2.imread(""),
        2: cv2.imread(""),
        3: cv2.imread(""),
        4: cv2.imread(""),
        5: cv2.imread(""),
    }
