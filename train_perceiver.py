import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import Perceiver
from dataset import NpyDataset, MixupNpyDataset

import yangdl as yd


parser = argparse.ArgumentParser(description='gen_features.py')
parser.add_argument('-t', '--t', type=str, help='encoder type', choices=['ctrans', 'vits16'])
args = parser.parse_args()
FEATURES_TYPE = args.t

yd.env.seed = 0
yd.env.exp_path = f'./res/{FEATURES_TYPE}_perceiver'

DATASET_PATH = './npy'
SPLIT_PATH = './split'

BATCH_SIZE = 4
NUM_CLASSES = 5
if FEATURES_TYPE == 'ctrans':
    FEATURES_DIM = 768
elif FEATURES_TYPE == 'vits16':
    FEATURES_DIM = 384

# mixup
MIXUP_ALPHA = 1.
SMOOTHING = 0.1
off_value = SMOOTHING / NUM_CLASSES
on_value = 1. - SMOOTHING + off_value


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def CrossEntropy(student_logits, teacher_logits):
    log_softmax_outputs = F.log_softmax(student_logits / 3.0, dim=1)
    softmax_targets = F.softmax(teacher_logits / 3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loss = yd.ValueMetric()
        self.metric = yd.ClsMetric(num_classes=NUM_CLASSES, properties=['recall', 'recalls'])

    def __iter__(self):
        for fold in range(1, 6):
            self.model = Perceiver(
                input_channels=FEATURES_DIM,
                input_axis=1,
                num_freq_bands=6,
                max_freq=10.,
                depth=1,
                num_latents=1024,
                latent_dim=FEATURES_DIM,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                n_classes=NUM_CLASSES,
                attn_dropout=0.2,
                ff_dropout=0.2,
                weight_tie_layers=True,
                fourier_encode_data=False,
                self_per_cross_attn=1,
                latent_bounds=2,
                scale=0.125,
            )

            self.optimizer = AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )

            yield

    def train_step(self, batch):
        loss_all = 0
        for b in batch:
            x, label, y = b[FEATURES_TYPE], b['label'], b['y']

            logits, _, _, _, results_dict = self.model(x)
            loss = self.criterion(logits, y).mean()

            # self-distillation
            f_t = torch.mean(results_dict['features_teacher'], 1).squeeze()
            f_s = results_dict['features_student'].squeeze()
            loss_coefficient = 0.3
            feature_loss_coefficient = 0.03
            loss += (self.criterion(results_dict['student_logits'], y) * (1 - loss_coefficient)).mean()
            loss += CrossEntropy(results_dict['student_logits'], logits) * loss_coefficient
            loss += torch.dist(f_s, f_t) * feature_loss_coefficient
            self.loss.update(loss, 1)
            loss /= BATCH_SIZE
            loss.backward()
            loss_all += loss

            probs = F.sigmoid(logits)
            self.metric.update(probs, label)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'loss': loss_all,
        }

    def val_step(self, batch):
        x, label = batch[FEATURES_TYPE][0], batch['label']

        logits, _, _, _, _ = self.model(x)
        
        y = F.one_hot(label, num_classes=NUM_CLASSES).float()
        loss = self.criterion(logits, y).mean()
        self.loss.update(loss, 1)

        probs = F.sigmoid(logits)
        self.metric.update(probs, label)

        return {
            'loss': loss,
            'bacc': self.metric.recall,
        }

    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
        )

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            bacc=self.metric.recall,
        )


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
            if isinstance(res, list):
                d0, d1 = res
                res = {}
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

                features0 = d0[FEATURES_TYPE]
                features1 = d1[FEATURES_TYPE]
                n0, n1 = len(features0), len(features1)
                maxn = max(n0, n1)
                features0 = np.pad(features0, ((0, maxn - n0), (0, 0)), mode='constant')
                features1 = np.pad(features1, ((0, maxn - n1), (0, 0)), mode='constant')

                res[FEATURES_TYPE] = torch.from_numpy(lam * features0 + (1 - lam) * features1)
                res['label'] = torch.tensor([d0['label'], d1['label']][lam >= 0.5])[None]
                res['y'] = lam * one_hot(torch.tensor(d0['label']), num_classes=NUM_CLASSES, on_value=on_value, off_value=off_value).float() + \
                    (1 - lam) * one_hot(torch.tensor(d1['label']), num_classes=NUM_CLASSES, on_value=on_value, off_value=off_value).float()

            return res

        for train_df in self.train_dfs:
            dataset = MixupNpyDataset(
                DATASET_PATH,
                train_df,
                transform=transform,
                rets=[FEATURES_TYPE, 'label'],
                cache=True,
            )
            yield DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True, drop_last=True, collate_fn=lambda x: x)

    def val_loader(self):
        def transform(res):
            res[FEATURES_TYPE] = torch.from_numpy(res[FEATURES_TYPE])
            res['label'] = torch.tensor(res['label'])

        for val_df in self.val_dfs:
            dataset = NpyDataset(
                DATASET_PATH,
                val_df,
                transform=transform,
                rets=[FEATURES_TYPE, 'label'],
                cache=True,
            )
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

