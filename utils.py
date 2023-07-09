import numpy as np
from typing import List

# ===================== Dict of colors ============================#

COLOR_LUT = {
    (20, 20, 20): 'BLACK',
    (200, 200, 200): 'WHITE',
    (255, 0, 0): 'RED',
    (169, 169, 169): 'GRAY',
    (0, 0, 255): 'BLUE',
    (0, 255, 0): 'GREEN',
    (21, 41, 69): 'DARK-BLUE'
}


# ===================Function that returns the color of the car by looking in the closest n neighbers ===========#


def classify_color(x: int, y: int, img: np.ndarray, n_size: int) -> str:
    """

    :param x:
    :param y:
    :param img:
    :param n_size:
    :return:
    """
    kx, ky = n_size
    neighborhood = img[y - ky // 2:y + ky // 2, x - kx // 2:x + kx // 2]  # extract neighborhood
    neighborhood = neighborhood.reshape(-1, neighborhood.shape[-1])  # flatten neighborhood
    neighborhood = (neighborhood // 10) * 10  # round to closest color

    color_dict = dict()
    # create color counter dictionary
    for color in neighborhood:
        color_dict[tuple(color)] = color_dict.get(tuple(color), 0) + 1

    # sort by values
    color_dict = {k: v for k, v in sorted(color_dict.items(), key=lambda item: item[1], reverse=True)}
    # The most frequent color found in the neighborhood
    most_fequent_color = list(color_dict.keys())[0]

    # Find closest color from color look up table
    LUT = list(COLOR_LUT.keys())
    distances_from_LUT = np.sqrt(np.sum((np.array(LUT) - most_fequent_color) ** 2, axis=1))
    index_of_min_distance = np.argmin(distances_from_LUT)

    classified_color = LUT[index_of_min_distance]
    classified_color_name = COLOR_LUT[classified_color]
    return classified_color_name


# ============================== Function that counts how many cars inside the roundbout ========================= #
def count_num_of_vehicles(pts: np.ndarray, mask: np.ndarray) -> int:
    """

    :param pts:
    :param mask:
    :return:
    """
    counter = 0

    for x0, y0 in pts[:, 0, :]:

        if x0 < 600 and y0 < 600 and mask[np.int(y0), np.int(x0)] > 0:
            counter += 1

    return counter


# ========================= Function that cleans 'good point to track' close to another 'good point to track'


def clean_duplicate(pts0: np.ndarray, pts1: np.ndarray) -> List:
    """

    :param pts0:
    :param pts1:
    :return:
    """
    threshold = 60
    P0 = pts0[:, 0, :]
    P1 = pts1[:, 0, :]
    result = []

    for x1, y1 in P0:

        Min = 99999

        for x11, y11 in P1:

            if (np.sqrt((x1 - x11) ** 2 + (y1 - y11) ** 2)) < Min:
                Min = np.sqrt((x1 - x11) ** 2 + (y1 - y11) ** 2)

        result.append(False) if Min < threshold else result.append(True)

    return result
