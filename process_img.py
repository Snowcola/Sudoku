import cv2
import numpy as np
import imutils
from utils import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s [%(levelname)s] : %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

default_image_location = Path.cwd() / "images"


class NoPuzzleFoundException(Exception):
    pass


def read_img(img_path, grayscale=True):
    if grayscale:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
    return img


# this needs refactoring
def find_puzzle(image, debug=False):
    # image = read_img("images\\sudoku.jpeg", grayscale=False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    threshhold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    threshhold = cv2.bitwise_not(threshhold)
    if debug:
        cv2.imshow("Puzzle", threshhold)
        cv2.waitKey(0)

    contours = cv2.findContours(
        threshhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )  # sort list, largets contours first
    puzzle_contour = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx_contour = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx_contour) == 4:  # pick the 4 sided contours
            puzzle_contour = approx_contour
            break

    if puzzle_contour is None:
        raise NoPuzzleFoundException(
            "Outline of the sudoku puzzle could not be found. Consider modifying threshold parameters"
        )

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)

        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    puzzle_image = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped_image = four_point_transform(gray, puzzle_contour.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle", puzzle_image)
        cv2.imshow("Warped", warped_image)
        # cv2.waitKey(0)
        cv2.waitKey(0)

    return (puzzle_image, warped_image)


def extract_digit(cell, debug=False):
    threshhold = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    threshhold = clear_border(threshhold)

    if debug:
        cv2.imshow("Cell Thresh", threshhold)
        cv2.waitKey(0)

    contours = cv2.findContours(
        threshhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(threshhold.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = threshhold.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(threshhold, threshhold, mask=mask)

    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit


def image_ratio(image, image_subsection):
    is_h, is_w, *_ = image_subsection.shape
    i_h, i_w, *_ = image.shape
    ratio = (is_h * is_w) / (i_h * i_w)
    logger.info(f"Puzzle/Image Ratio: {ratio:0.2f}")
    return ratio


def read_board(image, ocr_model, debug=False):
    (puzzle_image, warped) = find_puzzle(image, debug=debug)

    # naive approach at verifying if there is a puzzle
    if image_ratio(image, warped) < 0.2:
        raise NoPuzzleFoundException("Puzzle could not be found in image")

    cv2.imwrite(str(default_image_location / "warped.jpg"), warped)
    board = np.zeros((9, 9), dtype="int")

    cells = find_grid(warped, filter=True)

    for y, row in enumerate(cells):
        for x, cell in enumerate(row):
            leftX, topY, rightX, bottomY = cell
            cell_contents = warped[topY:bottomY, leftX:rightX]
            digit = extract_digit(cell_contents, debug=debug)

            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                pred = ocr_model.predict(roi).argmax(axis=1)[0]

                board[y, x] = pred

    logger.info(f"Board: \n{board}")
    logger.info(f"Total of cells on board: {np.sum(board)}")
    return board


def is_vertical_grid_line(p1, p2, threshhold=5):
    # only works on lines that are alreadt close to vertical or horizontal,
    x1, y1 = p1
    x2, y2 = p2
    if abs(x1 - x2) < threshhold:
        return True
    return False


def line_intersection(line1, line2):
    T = np.array([[0, -1], [1, 0]])
    a1, a2 = list(map(np.asarray, line1))
    b1, b2 = list(map(np.asarray, line2))

    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denominator = np.sum(dap * db, axis=1)
    numerator = np.sum(dap * dp, axis=1)
    return tuple((np.atleast_2d(numerator / denominator).T * db + b1).astype(int)[0])


def get_cells_from_points(verts):
    cells = np.zeros((9, 9, 4), np.uint32)

    for row in range(len(verts) - 1):
        for col in range(len(verts[row]) - 1):
            leftX, topY = verts[row][col]
            rightX, bottomY = verts[row + 1][col + 1]
            cell = (leftX, topY, rightX, bottomY)
            cells[row][col] = cell

    return cells


def find_grid(image, filter=False):
    edges = cv2.Canny(image, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    cv2.imwrite(str(default_image_location / "canny.jpg"), edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 145)

    if not lines.any():
        logger.error("No grid was found")
        return False, None

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue
                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if (
                    abs(rho_i - rho_j) < rho_threshold
                    and abs(theta_i - theta_j) < theta_threshold
                ):
                    similar_lines[i].append(j)

        indicies = [i for i in range(len(lines))]
        indicies.sort(key=lambda x: len(similar_lines[x]))

        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indicies[i]]:
                continue
            for j in range(i + 1, len(lines)):
                if not line_flags[indicies[j]]:
                    continue

                rho_i, theta_i = lines[indicies[i]][0]
                rho_j, theta_j = lines[indicies[j]][0]
                if (
                    abs(rho_i - rho_j) < rho_threshold
                    and abs(theta_i - theta_j) < theta_threshold
                ):
                    line_flags[indicies[j]] = False

    logger.info(f"Number of Hough Lines: {len(lines)}")

    filtered_lines = []

    if filter:
        for i in range(len(lines)):
            if line_flags[i]:
                filtered_lines.append(lines[i])
        logger.info(f"Number of filtered lines: {len(filtered_lines)}")
    else:
        filtered_lines = lines

    vertical_lines = []
    horizontal_lines = []
    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        grid_line = ((x1, y1), (x2, y2))

        if is_vertical_grid_line(grid_line[0], grid_line[1]):
            vertical_lines.append(grid_line)
        else:
            horizontal_lines.append(grid_line)

    intersections = []
    for h_line in horizontal_lines:
        row = []
        for v_line in vertical_lines:
            intersect = line_intersection(v_line, h_line)
            row.append(intersect)
        row.sort(key=lambda x: x[0])
        intersections.append(row)
    intersections.sort(key=lambda x: x[0][1])

    intersections = np.array(intersections, dtype="int")
    cells = get_cells_from_points(intersections)

    logger.info(f"H: {len(horizontal_lines)}")
    logger.info(f"V: {len(vertical_lines)}")
    logger.debug(vertical_lines)
    logger.debug(horizontal_lines)

    return cells


if __name__ == "__main__":
    image_path = Path("images") / "sudoku.jpeg"
    model = load_model(str(Path.cwd() / "ocr_output/new_model-2.h5"))
    img = read_img(str(image_path), grayscale=False)
    # find_puzzle(img, debug=True)
    read_board(img, ocr_model=model, debug=False)
    # image_path = Path(default_image_location) / "warped.jpg"
    # img = read_img(str(image_path), grayscale=False)
    # find_grid(img, filter=True)
