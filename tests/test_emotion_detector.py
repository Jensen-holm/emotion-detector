import sys
import os

sys.path.append(os.path.abspath(".."))

from models import EmotionDetector


def test_emotion_detector_init() -> None:
    """test the initialization of the EmotionDetector class"""
    emotion_model = EmotionDetector()

    assert emotion_model.emoji_map is not None
