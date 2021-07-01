import cv2
import numpy as np
import imutils
from utils import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array


class NoPuzzleFoundException(Exception):
    pass


def prepare_img(img):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    process = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
    )
    cv2.imwrite("images\\thresh.jpg", process)
    process = cv2.bitwise_not(process, process)
    cv2.imwrite("images\\bitwise.jpg", process)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
    process = cv2.dilate(process, kernel)
    cv2.imwrite("images\\all.jpg", process)
    return process


def read_img(img_path, grayscale=True):
    if grayscale:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
    return img


# this needs refactoring
def find_puzzle(img, debug=False):
    image = read_img("images\\sudoku.jpeg", grayscale=False)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    print(len(contours))
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
        threshhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
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


def test(image, debug=False):
    (puzzle_image, warped) = find_puzzle(image, debug=True)
    cv2.imwrite("images\\warped.jpg", warped)
    board = np.zeros((9, 9), dtype="int")

    cellSizeX = warped.shape[1] // 9
    cellSizeY = warped.shape[0] // 9

    # vertical lines
    for y in range(9):
        cv2.line(
            puzzle_image,
            (0, cellSizeY * y),
            (warped.shape[1], cellSizeY * y),
            (0, 0, 255),
            thickness=1,
        )

    cv2.imshow("lines", puzzle_image)
    cv2.waitKey(0)

    cellLocs = []

    for y in range(9):
        row = []
        for x in range(9):
            padding = 3

            leftX = max((x * cellSizeX) - padding, 0)
            rightX = min(((x + 1) * cellSizeX) + padding, 9 * cellSizeX)
            topY = max((y * cellSizeY) - padding, 0)
            bottomY = min(((y + 1) * cellSizeY) + padding, 9 * cellSizeY)

            row.append((leftX, topY, rightX, bottomY))
            cell = warped[topY:bottomY, leftX:rightX]
            # cv2.imshow("Cell", cell)
            # cv2.waitKey(0)
            digit = extract_digit(cell, debug=debug)

            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = 1  # pred
        # add the row to our cell locations
        cellLocs.append(row)
    print(*board, sep="\n")
    print(np.sum(board))


def show_image(img):
    cv2.imshow("image", prepare_img(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows


def extract_grid(img):
    pass


if __name__ == "__main__":
    img = read_img("images\\sudoku.jpeg")
    # show_image(img)
    # find_puzzle(img, debug=True)
    test(img, debug=False)
