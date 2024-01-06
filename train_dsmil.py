import argparse

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F

from model import DSMIL
from dataset import NpyDataset

import yangdl as yd


parser = argparse.ArgumentParser(description='gen_features.py')
parser.add_argument('-t', '--t', type=str, help='encoder type', choices=['ctrans', 'vits16'])
args = parser.parse_args()
FEATURES_TYPE = args.t

yd.env.seed = 0
yd.env.exp_path = f'./res/{FEATURES_TYPE}_dsmil'

DATASET_PATH = './npy'
SPLIT_PATH = './split'
BATCH_SIZE = 4
if FEATURES_TYPE == 'ctrans':
    SIZES = [768, 128, 128]
elif FEATURES_TYPE == 'vits16':
    SIZES = [384, 128, 128]


class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.loss = yd.ValueMetric()
        self.metric = yd.ClsMetric(num_classes=5, properties=['recall', 'recalls'])

    def __iter__(self):
        for fold in range(1, 6):
            self.model = DSMIL(
                num_classes=5,
                size=SIZES,
                dropout=0.5,
            )

            self.optimizer = AdamW(
                self.model.parameters(),
                lr=4e-4,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )

            yield

    def train_step(self, batch):
        loss_all = 0
        for b in batch:
            x, y = b[FEATURES_TYPE], b['label'][None]
            loss = self._step(x, y) / BATCH_SIZE
            loss.backward()
            loss_all += loss

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'loss': loss_all,
            'bacc': self.metric.recall,
        }

    def val_step(self, batch):
        x, y = batch[FEATURES_TYPE][0], batch['label']
        loss = self._step(x, y)

        return {
            'loss': loss,
            'bacc': self.metric.recall,
        }

    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            bacc=self.metric.recall,
        )

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            bacc=self.metric.recall,
        )

    # (N, C), (1,)
    def _step(self, x, y):
        bag_logits, inst_logits, _, _ = self.model(x)
        inst_logits, _ = torch.max(inst_logits, dim=0)
        bag_logits = bag_logits[None]
        inst_logits = inst_logits[None]

        bag_prob = F.softmax(bag_logits, dim=1)
        inst_prob = F.softmax(inst_logits, dim=1)
        prob = (bag_prob + inst_prob) / 2
        self.metric.update(prob, y)
        
        bag_loss = self.criterion(bag_logits, y).mean()
        inst_loss = self.criterion(inst_logits, y).mean()
        loss = (bag_loss + inst_loss) / 2
        self.loss.update(loss, 1)

        return loss


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

        self.train_dfs = []
        self.val_dfs = []
        for fold in range(1, 6):
            self.train_dfs.append(pd.read_csv(f'{SPLIT_PATH}/train/{fold}.csv', index_col='file_name', dtype={'file_name': str}))
            self.val_dfs.append(pd.read_csv(f'{SPLIT_PATH}/val/{fold}.csv', index_col='file_name', dtype={'file_name': str}))

    def train_loader(self):
        def transform(res):
            res[FEATURES_TYPE] = torch.tensor(res[FEATURES_TYPE])
            res['label'] = torch.tensor(res['label'])

        for train_df in self.train_dfs:
            dataset = NpyDataset(
                DATASET_PATH,
                train_df,
                transform=transform,
                rets=[FEATURES_TYPE, 'label'],
                cache=True,
            )
            print(len(dataset))
            yield DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True, drop_last=True, collate_fn=lambda x: x)

    def val_loader(self):
        def transform(res):
            res[FEATURES_TYPE] = torch.tensor(res[FEATURES_TYPE])
            res['label'] = torch.tensor(res['label'])

        for val_df in self.val_dfs:
            dataset = NpyDataset(
                DATASET_PATH,
                val_df,
                transform=transform,
                rets=[FEATURES_TYPE, 'label'],
                cache=True,
            )
            print(len(dataset))
            yield DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

if __name__ == '__main__':
    task_module = yd.TaskModule(
        model_module=MyModelModule(),
        data_module=MyDataModule(),
        early_stop_params={
            'monitor': {'metric.recall': 'max'},
            'patience': 25,
            'min_stop_epoch': 25,
            'max_stop_epoch': 200,
        },
    )
    res = task_module.do()

    print(f"balanced accuracy: {res['val']['metric']['recall']:.4f}")

