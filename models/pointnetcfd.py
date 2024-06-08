"""
Contains classed comprising Point Net Architecture. Usage for each class can 
be found in main() at the bottom.

TO use: Import Classification and Segmentation classes into desired script



NOTE:
This architecture does not cover Part Segmentation. Per the Point Net paper 
that is a different architecture and is not implemented here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALING_FACTOR = 1


# ============================================================================
# T-net (Spatial Transformer Network)
class Tnet(nn.Module):
    """T-Net learns a Transformation matrix with a specified dimension"""

    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(1024, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)
        self.act = nn.Tanh()

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(self.act(self.conv1(x)))
        x = self.bn2(self.act(self.conv2(x)))
        x = self.bn3(self.act(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)

        # pass through MLP
        x = self.bn4(self.act(self.linear1(x)))
        x = self.bn5(self.act(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x


# ============================================================================
# Point Net Backbone (main Architecture)
class PointNetBackbone(nn.Module):
    """
    This is the main portion of Point Net before the classification and segmentation heads.
    The main function of this network is to obtain the local and global point features,
    which can then be passed to each of the heads to perform either classification or
    segmentation. The forward pass through the backbone includes both T-nets and their
    transformations, the shared MLPs, and the max pool layer to obtain the global features.

    The forward function either returns the global or combined (local and global features)
    along with the critical point index locations and the feature transformation matrix. The
    feature transformation matrix is used for a regularization term that will help it become
    orthogonal. (i.e. a rigid body transformation is an orthogonal transform and we would like
    to maintain orthogonality in high dimensional space). "An orthogonal transformations preserves
    the lengths of vectors and angles between them"
    """

    def __init__(
        self, num_points=2500, num_global_feats=1024, local_feat=True, use_tnet=False
    ):
        """Initializers:
        num_points - number of points in point cloud
        num_global_feats - number of Global Features for the main
                           Max Pooling layer
        local_feat - if True, forward() returns the concatenation
                     of the local and global features
        """
        super(PointNetBackbone, self).__init__()

        # if true concat local and global features
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat
        self.use_tnet = use_tnet

        if self.use_tnet:
            # Spatial Transformer Networks (T-nets)
            self.tnet1 = Tnet(dim=2, num_points=num_points)
            self.tnet2 = Tnet(dim=64, num_points=num_points)

        # shared MLP 1
        self.conv1 = nn.Conv1d(2, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        # batch norms for both shared MLPs
        # self.bn1 = nn.Identity()  # nn.BatchNorm1d(64, track_running_stats=False)
        # self.bn2 = nn.Identity()  # nn.BatchNorm1d(64, track_running_stats=False)
        # self.bn3 = nn.Identity()  # nn.BatchNorm1d(64, track_running_stats=False)
        # self.bn4 = nn.Identity()  # nn.BatchNorm1d(128, track_running_stats=False)
        # self.bn5 = nn.Identity()
        self.bn = nn.Identity()
        # nn.BatchNorm1d(self.num_global_feats, track_running_stats=False)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)
        self.act = nn.Tanh()

    def forward(self, x):

        # get batch size
        bs = x.shape[0]

        if self.use_tnet:
            # pass through first Tnet to get transform matrix
            A_input = self.tnet1(x)

            # perform first transformation across each point in the batch
            x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        # pass through first shared MLP
        x = self.bn(self.act(self.conv1(x)))
        x = self.bn(self.act(self.conv2(x)))

        if self.use_tnet:
            # get feature transform
            A_feat = self.tnet2(x)

            # perform second transformation across each (64 dim) feature in the batch
            x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        else:
            A_feat = None

        # store local point features for segmentation head
        local_features = x.clone()

        # pass through second MLP
        x = self.bn(self.act(self.conv3(x)))
        x = self.bn(self.act(self.conv4(x)))
        x = self.bn(self.act(self.conv5(x)))

        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat(
                (
                    local_features,
                    global_features.unsqueeze(-1).repeat(1, 1, self.num_points),
                ),
                dim=1,
            )

            return features, critical_indexes, A_feat

        else:
            return global_features, critical_indexes, A_feat


# ============================================================================
# Segmentation Head
class PointNetSegHead(nn.Module):
    """Segmentation Head"""

    def __init__(self, num_points=2500, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        # get the backbone
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)

        # shared MLP
        num_features = num_global_feats + 64  # local and global features
        self.conv1 = nn.Conv1d(num_features, int(512 * SCALING_FACTOR), kernel_size=1)
        self.conv2 = nn.Conv1d(
            int(512 * SCALING_FACTOR), int(256 * SCALING_FACTOR), kernel_size=1
        )
        self.conv3 = nn.Conv1d(
            int(256 * SCALING_FACTOR), int(128 * SCALING_FACTOR), kernel_size=1
        )
        self.conv4 = nn.Conv1d(int(128 * SCALING_FACTOR), m, kernel_size=1)

        # batch norms for shared MLP
        # self.bn1 = nn.Identity()  # nn.BatchNorm1d(512, track_running_stats=False)
        # self.bn2 = nn.Identity()  # nn.BatchNorm1d(256, track_running_stats=False)
        # self.bn3 = nn.Identity()  # nn.BatchNorm1d(128, track_running_stats=False)
        # self.activ = torch.nn.Sigmoid()
        self.bn = nn.Identity()
        self.act = nn.Tanh()

        self.initial_param()

    def native_forward(self, x):

        # get combined features
        x, crit_idxs, A_feat = self.backbone(x)

        # pass through shared MLP
        x = self.bn(self.act(self.conv1(x)))
        x = self.bn(self.act(self.conv2(x)))
        x = self.bn(self.act(self.conv3(x)))
        x = self.conv4(x)

        # x = self.act(x)
        return x, crit_idxs, A_feat

    def forward(self, x):
        assert x.shape[1] == 2
        _modified = False
        if x.shape[-1] < self.num_points:
            _modified = True
            old_len = x.shape[-1]
            additional_length = self.num_points - old_len
            zero_tensor = torch.ones(x.shape[0], x.shape[1], additional_length).to(
                x.device
            )
            x = torch.cat([x, zero_tensor], dim=-1)

        # get combined features
        x, crit_idxs, A_feat = self.backbone(x)

        # pass through shared MLP
        x = self.bn(self.act(self.conv1(x)))
        x = self.bn(self.act(self.conv2(x)))
        x = self.bn(self.act(self.conv3(x)))
        x = self.conv4(x)

        # x = self.act(x)
        if _modified:
            x = x[:, :, :old_len]
        return x

    def initial_param(self):
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                try:
                    nn.init.xavier_normal_(param)
                except:
                    print("error: ", name)
            elif name.endswith("bias"):
                nn.init.normal_(param)


# ============================================================================
# Test
def main():
    ## test segmentation head
    test_data = torch.rand(32, 2, 400)

    seg = PointNetSegHead(m=3)
    out, _, _ = seg(test_data)
    print(out)
    print(f"Seg shape: {out.shape}")


if __name__ == "__main__":
    main()
