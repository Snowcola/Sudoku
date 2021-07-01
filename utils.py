import cv2
import numpy as np


def order_points(points: np.array) -> np.array:
    """orders an array of points in the order of: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    sum_points = points.sum(axis=1)
    diff_points = np.diff(points, axis=1)

    rect[0] = points[np.argmin(sum_points)]  # top left
    rect[2] = points[np.argmax(sum_points)]  # bottom right
    rect[1] = points[np.argmin(diff_points)]  # top right
    rect[3] = points[np.argmax(diff_points)]  # bottom left

    return rect


def four_point_transform(img, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    bottomWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(topWidth), int(bottomWidth))

    rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(leftHeight), int(rightHeight))

    dest = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dest)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


if __name__ == "__main__":
    pass
