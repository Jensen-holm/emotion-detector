import cv2

from models import FaceDetector, EmotionDetector

CV_QUIT_KEYS: set[int] = {27, ord("q")}  # ESC || q


def main(cam_idx: int) -> None:
    """opens video capture at cam_idx, looks for faces and detects emotions in real time"""

    cap = cv2.VideoCapture(cam_idx)
    if not (ok := cap.isOpened()):
        raise Exception(
            f"webcam was found at idx {cam_idx}, but was unable to be opened"
        )

    emotion_model = EmotionDetector()
    face_model = FaceDetector()
    while ok:
        ret, frame = cap.read()
        if not ret:
            break

        for face in face_model.predict(frame):
            cropped_face = face.crop(frame)
            emoji = emotion_model.predict(cropped_face)
            face.overlay(frame, emoji)

        cv2.imshow("emotion detection", frame)

        if cv2.waitKey(1) in CV_QUIT_KEYS:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main(cam_idx=0)
