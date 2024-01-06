import numpy as np
import pandas as pd
import albumentations as A

from yangUtils.io import read_png
from utils import (
    deduplicate,
    image2patches,
    mkdir,
    read_png,
)


OUT_PATCHES_PATH = './npy/patches'
OUT_COORDS_PATH = './npy/coords'
mkdir(OUT_PATCHES_PATH, OUT_COORDS_PATH)


if __name__ == '__main__':
    df = pd.read_csv(f'./label.csv', index_col='file_name', dtype={'file_name': str})
    file_names = df.index
    for i, file_name in enumerate(file_names, 1):
        print(f'{i}/{len(file_names)}, {file_name}:')

        image = read_png(f'./train_images/{file_name}.png')

        # 1. downsample
        is_tma = df.loc[file_name, 'is_tma']
        if is_tma:
            resize = A.Resize(image.shape[0] // 4, image.shape[1] // 4)
        else:
            resize = A.Resize(image.shape[0] // 2, image.shape[1] // 2)
        image = resize(image=image)['image']

        # 2. deduplicate
        if not is_tma:
            image = deduplicate(image)

        # 3. image2patches
        patches, coords = image2patches(image, 256, [256, 64][int(is_tma)], 0.25, is_tma)

        np.save(f'{OUT_PATCHES_PATH}/{file_name}.npy', patches)
        np.save(f'{OUT_COORDS_PATH}/{file_name}.npy', coords)

        print(patches.shape, coords.shape)

