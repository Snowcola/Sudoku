from sudoku_solver.camera import get_frame_valid, show_puzzle_area_in_frame
from sudoku_solver.process_img import image_ratio, extract_possible_grid
import numpy as np


def area_validator(frame: np.ndarray, min_puzzle_area: float) -> bool:
    _, puzzle = extract_grid(frame)
    if image_ratio(frame, puzzle) > min_puzzle_area:
        return True
    return False


if __name__ == "__main__":
    # get_frame_valid(area_validator, min_puzzle_area=0.2)
    show_puzzle_area_in_frame(scale=0.5)
