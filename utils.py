import math
import os

import albumentations as A
import cv2
import numpy as np
import pyvips


__all__ = [
    'get_file_names',
    'mkdir',
    'rgb2gray',
    'deduplicate',
    'image2patches',
]


def get_file_names(path: str, suffix: str) -> list[str]:
    file_names = os.listdir(path)
    file_names = [(file_name[:-len(suffix)]) for file_name in file_names if file_name[-len(suffix):] == suffix]
    return file_names


def mkdir(*args: str) -> None:
    for dir_name in args:
        if dir_name is None:
            continue
        if '.' in os.path.basename(dir_name):  # if is file
            dir_name = os.path.dirname(dir_name)
        if dir_name == '':
            continue
        os.makedirs(dir_name, exist_ok=True)


def read_png(file_name: str):
    return pyvips.Image.new_from_file(file_name, access='sequential').numpy()


def rgb2gray(image: np.ndarray):
    image = image.astype(np.float16)
    image = (image[..., 0] * 299 + image[..., 1] * 587 + image[..., 2] * 114) / 1000

    return image.astype(np.uint8)


def deduplicate(image):
    resize = A.Resize(image.shape[0] // 16, image.shape[1] // 16)  # downsample for speed
    thumbnail = resize(image=image)['image']
    mask = rgb2gray(thumbnail) > 0
    x0, y0, x1, y1 = get_biggest_component_box(mask)

    # resize box
    scale_h = image.shape[0] / thumbnail.shape[0]
    scale_w = image.shape[1] / thumbnail.shape[1]

    x0 = max(0, math.floor(x0 * scale_w))
    y0 = max(0, math.floor(y0 * scale_h))
    x1 = min(image.shape[1] - 1, math.ceil(x1 * scale_w))
    y1 = min(image.shape[0] - 1, math.ceil(y1 * scale_h))

    return image[y0: y1 + 1, x0: x1 + 1]


def get_biggest_component_box(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    max_idx = np.argmax(areas)

    contour = contours[max_idx]

    x0 = contour[:, 0, 0].min()
    y0 = contour[:, 0, 1].min()
    x1 = contour[:, 0, 0].max()
    y1 = contour[:, 0, 1].max()

    return x0, y0, x1, y1


def image2patches(image: np.ndarray, patch_size: int, step: int, ratio: float, is_tma: bool = False):
    """
    Args:
        image (H, W, 3)

    Returns:
        patches: (N, 256, 256, 3), np.uint8
        coords: (N, 2), np.int64
    """
    patches = []
    coords = []
    for i in range(0, image.shape[0], step):
        for j in range(0, image.shape[1], step):
            patch = image[i: i + patch_size, j: j + patch_size, :]
            if patch.shape != (patch_size, patch_size, 3):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)))

            if is_tma:
                patches.append(patch)
                coords.append((i, j))
            else:
                patch_gray = rgb2gray(patch)
                patch_binary = (patch_gray <= 220) & (patch_gray > 0)

                if np.count_nonzero(patch_binary) / patch_binary.size >= ratio:
                    patches.append(patch)
                    coords.append((i, j))

    patches = np.array(patches)
    coords = np.array(coords)

    return patches, coords
