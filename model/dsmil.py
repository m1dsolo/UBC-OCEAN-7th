import torch
from torch import nn
from torch.nn import functional as F


class DSMIL_Attention(nn.Module):
    def __init__(self):
        super().__init__()

    # q(patch_num, size[2]), q_max(num_classes, size[2])
    def forward(self, q, q_max):
        attn = q @ q_max.transpose(1, 0) # (patch_num, num_classes)

        return F.softmax(attn / torch.sqrt(torch.tensor(q.shape[1], dtype=torch.float32)), dim=0) # (patch_num, num_classes)


class DSMIL_BClassifier(nn.Module):
    def __init__(self,
        num_classes: int,
        size = [768, 128, 128],
        dropout: float = 0.5,
    ):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Linear(size[1], size[2]),
            nn.Tanh()
        )
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(size[0], size[0]),
            nn.ReLU()
        )
        self.attention = DSMIL_Attention()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_classes, num_classes, kernel_size=size[0])
        )

    # x(patch_num, size[0]), inst_logits(patch_num, num_classes)
    def forward(self, x, inst_logits):
        v = self.v(x)  # (patch_num, size[0])
        q = self.q(x)  # (patch_num, size[2])

        _, idxs = torch.sort(inst_logits, dim=0, descending=True)  # (patch_num, num_classes)
        idxs = idxs[0]   # (num_classes,)
        x_sub = x[idxs]   # (num_classes, size[0])
        q_max = self.q(x_sub)  # (num_classes, size[2])

        attn = self.attention(q, q_max)  # (patch_num, num_classes)
        
        bag_feature = attn.transpose(1, 0) @ v # (num_classes, size[0])
        bag_logits = self.classifier(bag_feature)[:, 0] # (num_classes,)

        return bag_logits, attn, bag_feature


class DSMIL(nn.Module):
    def __init__(self,
        num_classes: int,
        size = [768, 128, 128],
        dropout: float = 0.5,
    ):
        super().__init__()

        self.i_classifier = nn.Linear(size[0], num_classes)
        self.b_classifier = DSMIL_BClassifier(num_classes, size, dropout)

    def forward(self, x):
        inst_logits = self.i_classifier(x)
        bag_logits, attn, bag_feature = self.b_classifier(x, inst_logits)

        # (num_classes,), (N, num_classes), 
        return bag_logits, inst_logits, attn, bag_feature

