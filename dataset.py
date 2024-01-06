from collections import defaultdict
import os
import random
import resource
from typing import Optional, Callable

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import get_file_names


class NpyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        rets: list[str] = ['image'],
        *,
        mmap_mode: Optional[str] = None,
        cache: bool = False,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.df = df
        self.transform = transform
        self.file_names = []
        if df is not None:
            file_names = df.index.tolist()
        else:
            file_names = get_file_names(f'{dataset_path}/{rets[0]}', '.npy')
            if len(file_names) == 0:
                file_names = get_file_names(f'{dataset_path}/{rets[0]}')
        for file_name in file_names:
            if os.path.isdir(f'{dataset_path}/{rets[0]}/{file_name}'):
                sub_file_names = get_file_names(f'{dataset_path}/{rets[0]}/{file_name}', '.npy')
                for sub_file_name in sub_file_names:
                    self.file_names.append(f'{file_name}/{sub_file_name}')
            elif os.path.exists(f'{dataset_path}/{rets[0]}/{file_name}.npy'):
                self.file_names.append(file_name)
            else:
                raise FileNotFoundError(f'Cant find {dataset_path}/{rets[0]}/{file_name} or {dataset_path}/{rets[0]}/{file_name}.npy')

        self.rets = rets

        self.mmap_mode = mmap_mode
        if mmap_mode == 'r':
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        self.cache = cache
        if self.cache:
            self.cache_dict = defaultdict(dict)
            for file_name in tqdm(self.file_names, desc='Dataset loading: '):
                for ret in self.rets:
                    if os.path.exists(f'{dataset_path}/{ret}/{file_name}.npy'):
                        self.cache_dict[file_name][ret] = np.load(f'{dataset_path}/{ret}/{file_name}.npy', mmap_mode=self.mmap_mode)

    def __len__(self):
        return len(self.file_names)

    def get_one_item(self, idx):
        file_name = self.file_names[idx]
        res = {}
        for ret in self.rets:
            if os.path.exists(f'{self.dataset_path}/{ret}/{file_name}.npy'):
                res[ret] = self.cache_dict[file_name][ret] if self.cache else np.load(f'{self.dataset_path}/{ret}/{file_name}.npy', mmap_mode=self.mmap_mode)
            elif ret == 'file_name':
                res['file_name'] = file_name
            elif file_name in self.df.index and ret in self.df:
                res[ret] = self.df.loc[file_name, ret]
            elif file_name.split('/')[0] in self.df.index and ret in self.df:
                res[ret] = self.df.loc[file_name.split('/')[0], ret]
            else:
                res[ret] = None

        return res

    def __getitem__(self, idx):
        res = self.get_one_item(idx)
        self.transform(res)

        return res


class MixupNpyDataset(NpyDataset):
    def __init__(
        self,
        dataset_path: str,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        rets: list[str] = ['image'],
        *,
        mmap_mode: Optional[str] = None,
        cache: bool = False,
        mixup_prob: float = 1.,
    ):
        super().__init__(dataset_path, df, transform, rets, mmap_mode=mmap_mode, cache=cache)

        self.mixup_prob = mixup_prob

    def __getitem__(self, idx):
        if self.mixup_prob >= random.random():
            res = [self.get_one_item(idx), self.get_one_item(random.randint(0, len(self) - 1))]
            res = self.transform(res)
        else:
            res = self.transform(self.get_one_item(idx))

        return res

