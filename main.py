from cv2.typing import MatLike
from typing import Tuple
import time
import cv2

from models import FaceDetector, EmotionDetector
from models.face_detector.face_info import FaceInfo

CV_QUIT_KEYS: set[int] = {27, ord("q")}  # ESC || q

EMOJI_LOCS: list[Tuple[int, int, int, int]] = [
    (i, 0, i + 150, 150) for i in range(0, 1000, 200)
]


def main(cam_idx: int, refresh_interval: float) -> None:
    """opens video capture at cam_idx, looks for faces and detects emotions in real time"""
    cap = cv2.VideoCapture(cam_idx)
    if not (ok := cap.isOpened()):
        raise Exception(
            f"webcam was found at idx {cam_idx}, but was unable to be opened"
        )
    cv2.namedWindow("Emotion Detector", cv2.WND_PROP_FULLSCREEN)

    emotion_model = EmotionDetector()
    face_model = FaceDetector()

    faces: dict[int, MatLike] = {}
    last_refresh = time.time()
    while ok:
        ret, frame = cap.read()
        if not ret:
            break

        pred_faces = face_model.predict(frame)
        pred_faces = sorted(
            pred_faces, key=lambda face: face.x1
        )  # tracking faces using left-right positioning

        cur_time = time.time()
        if (cur_time - last_refresh) >= refresh_interval or not faces:
            faces.clear()
            for face_num, face in enumerate(pred_faces):
                emoji = emotion_model.predict(face.crop(frame))
                faces[face_num] = emoji
            last_refresh = cur_time

        for face_num, face in enumerate(pred_faces):
            emoji = faces.get(face_num, None)
            if emoji is None:
                continue
            face.overlay(frame, emoji)

        # resize frame to full screen size (might have to adjust this value once we have monitor)
        frame = cv2.resize(frame, (2200, 1080), interpolation=cv2.INTER_LINEAR)
        # display available emojis on the screen
        for emoji, loc in zip(emotion_model.emoji_map.values(), EMOJI_LOCS):
            FaceInfo(*loc).overlay(frame, emoji) if loc else 0

        cv2.imshow("Emotion Detector", frame)
        if (
            cv2.waitKey(1) in CV_QUIT_KEYS
            or cv2.getWindowProperty("Emotion Detector", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main(cam_idx=0, refresh_interval=0.5)
