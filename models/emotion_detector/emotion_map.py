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

    angry = __load_emoji("angry_emoji.png")
    smiling = __load_emoji("smiling_emoji.png")
    sad = __load_emoji("sad_emoji.png")
    surprised = __load_emoji("surprised_emoji.png")
    neutral = __load_emoji("neutral_emoji.png")
    return {
        0: angry,
        # 1: sad, # disgust
        # 2: angry, # fear
        3: smiling,
        4: sad,
        5: surprised,
        6: neutral,
    }
