import cv2


def use_cam_example(scale: int = 1) -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Input", frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    use_cam_example(scale=0.3)
