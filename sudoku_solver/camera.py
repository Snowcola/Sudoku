import cv2
from numpy import ndarray
from typing import Callable


def get_frame_valid(
    validator: Callable[[ndarray], bool],
    scale: int = 1,
) -> ndarray:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Input", frame)
        if validator(frame):
            cap.release()
            cv2.destroyAllWindows()
            return frame

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_frame_valid(lambda x: False, scale=0.3)
