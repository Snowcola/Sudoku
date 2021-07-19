import cv2
import logging
from numpy import ndarray
from typing import Callable
from .process_img import (
    find_puzzle_contour,
    draw_puzzle_contour,
    pre_process_image,
    extract_possible_grid,
    find_grid,
    locate_grid_lines,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s [%(levelname)s] : %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_frame_valid(
    validator: Callable[[ndarray], bool], scale: float = 1.0, **kwargs
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
        if validator(frame, **kwargs):
            cap.release()
            cv2.destroyAllWindows()
            return frame

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_lines(image, lines, color=(0, 0, 255)):
    logger.info(f"{lines=}")
    logger.info(f"{len(lines)=}")
    for line in lines:
        logger.info(line)
        cv2.line(image, line[0], line[1], color)
    return image


def show_puzzle_area_in_frame(scale: float = 1.0):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

        _, threshhold = pre_process_image(frame)
        puzzle_zone = find_puzzle_contour(threshhold)
        color_grid, gray_grid = extract_possible_grid(
            frame
        )  # TODO: refactor this duplicates calcs
        lines = locate_grid_lines(gray_grid, filter=True)
        logger.info(len(lines))
        # cells = find_grid(gray_grid, filter=True)
        # if cells and cells.shape == (9, 9, 4):
        #   pass
        frame = draw_puzzle_contour(frame, puzzle_zone)
        if len(lines) >= 20:
            color_grid = draw_lines(color_grid, lines)
        cv2.imshow("grid", color_grid)
        cv2.imshow("Input", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_frame_valid(lambda x: False, scale=0.3)
