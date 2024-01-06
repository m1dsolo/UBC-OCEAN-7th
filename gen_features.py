import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from dataset import NpyDataset
from model import CTransPath
from utils import mkdir
import yangdl as yd


parser = argparse.ArgumentParser(description='gen_features.py')
parser.add_argument('-t', '--t', type=str, help='encoder type', choices=['ctrans', 'vits16'])
args = parser.parse_args()
FEATURES_TYPE = args.t

yd.env.seed = 0

OUT_PATH = f'./npy/features/{FEATURES_TYPE}'
mkdir(OUT_PATH)

DATASET_PATH = f'./npy/256'
IMAGE_PATCH_NAME = 'patches'
BS = 64


class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()
        
        if FEATURES_TYPE == 'ctrans':
            self.model = CTransPath(num_classes=0)
            self.model.load_state_dict(torch.load(f'./ctranspath.pth')['model'])
        elif FEATURES_TYPE == 'vits16':
            self.model = timm.create_model(
                model_name="hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
                pretrained=True,
            )

    def predict_step(self, batch):
        x, file_name = batch[IMAGE_PATCH_NAME][0], batch['file_name'][0]

        features = []
        for i in range(0, len(x), BS):
            features.append(self.model(x[i: i + BS]).cpu().numpy())
        features = np.concatenate(features, axis=0)

        np.save(f'{OUT_PATH}/{file_name}.npy', features)


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

    def predict_loader(self):
        trans = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.815, 0.695, 0.808), std=(0.129, 0.147, 0.112)),
            ToTensorV2(),
        ])
        def transform(res):
            image_patches = []
            for image_patch in res[IMAGE_PATCH_NAME]:
                image_patches.append(trans(image=image_patch)['image'])
            res[IMAGE_PATCH_NAME] = torch.stack(image_patches, dim=0)

        dataset = NpyDataset(
            DATASET_PATH,
            transform=transform,
            rets=[IMAGE_PATCH_NAME, 'file_name'],
        )
        print(len(dataset))
        yield DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=False, pin_memory=False)


if __name__ == '__main__':
    task_module = yd.TaskModule(model_module=MyModelModule(), data_module=MyDataModule())
    task_module.do()

