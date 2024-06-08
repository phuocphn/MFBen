from __future__ import print_function

import torch
import numpy as np
import random
import glob

import wandb
import hydra
import os
from tqdm import tqdm
from pathlib import Path
from models import mlp
from models import pointnetcfd
from models import mlp_conv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd as autograd
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from pinnstorch.data.domains.point_cloud import PointData

# from pinnstorch.data import GeometryDataLoader
from omegaconf import DictConfig, OmegaConf
from visualize import visualize, visualize_single

device = "cuda" if torch.cuda.is_available() else "cpu"
# ensure reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, use_max_fn=True, use_bn=False):
        super(PointNetfeat, self).__init__()
        self.use_max_fn = use_max_fn
        self.use_bn = use_bn
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv11 = nn.Conv1d(64, 64, 1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()
        # self.maxfn = torch.nn.AdaptiveAvgPool1d(1)
        self.global_feat = global_feat
        self.weight_init(self.conv1)
        self.weight_init(self.conv2)
        self.weight_init(self.conv3)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        if self.use_bn:
            x = self.act1(self.bn1(self.conv1(x)))
            pointfeat = x
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.act3(self.bn3(self.conv3(x)))
        else:
            x = self.act1(self.conv1(x))
            x = self.act1(self.conv11(x))

            pointfeat = x
            x = self.act2(self.conv2(x))
            x = self.act3(self.conv3(x))

        if self.use_max_fn:
            x = torch.max(x, 2, keepdim=True)[0]
            return torch.cat([x.repeat(1, 1, n_pts), pointfeat], 1)
        else:
            return torch.cat([x, pointfeat], 1)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
            # print(f"Initialized weight for: {m}")


class PointNetSeg(nn.Module):
    def __init__(self, **kwargs):
        super(PointNetSeg, self).__init__()

        use_max_fn = kwargs.get("use_max_fn", True)
        use_bn = kwargs.get("use_bn", False)
        use_transpose = kwargs.get("use_transpose", False)

        self.use_bn = use_bn
        self.use_transpose = use_transpose
        print(
            "use_max_fn", use_max_fn, "use_bn", use_bn, "use_transpose", use_transpose
        )
        self.k = 3
        self.feat = PointNetfeat(
            global_feat=False, use_max_fn=use_max_fn, use_bn=use_bn
        )
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)

        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()

        self.weight_init(self.feat)
        self.weight_init(self.conv1)
        self.weight_init(self.conv2)
        self.weight_init(self.conv3)
        self.weight_init(self.conv4)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)

        if self.use_bn:
            x = self.act1(self.bn1(self.conv1(x)))
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.act3(self.bn3(self.conv3(x)))
        else:
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            x = self.act3(self.conv3(x))

        x = self.conv4(x)
        if self.use_transpose:
            x = x.transpose(2, 1).contiguous()
            x = F.sigmoid(x.view(-1, self.k), dim=-1)
            x = x.view(batchsize, self.k, n_pts)
        return x

    def weight_init(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
            # print(f"Initialized weight for: {m}")


import torch
import torch.nn as nn
import torch.nn.init as init


class MLPConv(nn.Module):
    """
    Neural network model consisting of MLP layers.
    """

    def __init__(
        self, hidden_layers: int = 20, layer_neurons: int = 32, num_outputs: int = 3
    ):
        """Define and allocate layers for this neural net.

        :param hidden_layers: number of hidden layers in this neural net, defaults to 20
        :param layer_neurons: the number of neurons of each layer, defaults to 32
        :param num_outputs: the number of output variables, defaults to 3
        """

        super().__init__()

        # Input size should be : (num_cells, 2)
        self.hidden_layers = hidden_layers
        self.layer_neurons = layer_neurons
        self.num_outputs = num_outputs
        dimension = 2
        layer_mat = [dimension] + hidden_layers * [layer_neurons] + [num_outputs]

        self.net = self._build_net(layer_mat)
        self.net.apply(self.weight_init)

    def _build_net(self, layer_mat, add_last_activation: bool = False):
        net = torch.nn.Sequential()
        layer_num = len(layer_mat) - 1
        for i in range(0, layer_num - 1):
            # net.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            net.add_module(
                str(i) + "linear",
                nn.Conv1d(layer_mat[i], layer_mat[i + 1], kernel_size=1),
            )
            net.add_module(str(i) + "act", nn.Tanh())

        # net.add_module(f"linear-last", nn.Linear(layer_mat[i + 1], layer_mat[i + 2]))
        net.add_module(
            f"linear-last", nn.Conv1d(layer_mat[i + 1], layer_mat[i + 2], kernel_size=1)
        )

        if add_last_activation:
            net.add_module("output-act", nn.Tanh())
        return net

    def weight_init(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
            # print(f"Initialized weight for: {m}")

    def forward(self, x):
        """Pass inputs to the self.net to obtain the outputs

        :param x: Input tensor
        :return: Output tensor, with the shape (num_cells, self.num_outputs)
        """
        x = self.net(x)
        return x


# ======================

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.Tanh()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 4)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 2, 2)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.Tanh()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat_v2(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat_v2, self).__init__()
        self.stn = STNkd(k=2)
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.tanh(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.tanh(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_v2(
            global_feat=True, feature_transform=feature_transform
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.Tanh()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.tanh(self.bn1(self.fc1(x)))
        x = F.tanh(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, **kwargs):
        super(PointNetDenseCls, self).__init__()
        k = 3
        feature_transform = False
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_v2(
            global_feat=False, feature_transform=feature_transform
        )
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        import pudb

        # pu.db
        x = torch.permute(x, (0, 2, 1))
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss
