from cv2.typing import MatLike
import cv2
import os


def load_emoji_map() -> dict[int, MatLike]:
    """
    loads a dictionary of the emoji's in ./assets that correspond to
    the emotion detection models output.
    details: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv
    """

    def __load_emoji(file_name: str) -> MatLike:
        path = os.path.join(
            os.path.dirname(__file__),
            "emojis",
            file_name,
        )
        return cv2.imread(path)

    return {
        0: __load_emoji("angry_emoji.png"),
        1: __load_emoji("disgust_emoji.png"),
        2: __load_emoji("scared_emoji.png"),
        3: __load_emoji("smiling_emoji.png"),
        4: __load_emoji("sad_emoji.png"),
        5: __load_emoji("surprised_emoji.png"),
        6: __load_emoji("neutral_emoji.png"),
    }
