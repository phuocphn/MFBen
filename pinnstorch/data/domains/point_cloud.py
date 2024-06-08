import torch
from dataclasses import dataclass


class PointData:

    def __init__(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        u: torch.Tensor = None,
        v: torch.Tensor = None,
        p: torch.Tensor = None,
    ):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.p = p

        k = 512
        self.fixed_indices = torch.randperm(self.x.shape[-1])[:k]
        self._update_k = False
        print(f"There are {x.shape[-1]} collocation points")

    # def construct_inputs(self):
    #     x = torch.tensor(self.x, dtype=torch.float32, requires_grad=True)
    #     y = torch.tensor(self.y, dtype=torch.float32, requires_grad=True)
    #     inputs = torch.hstack([x, y]).requires_grad_(True)
    #     return inputs, x, y

    # def construct_targets(self):
    #     targets = torch.hstack([self.u, self.v, self.p])
    #     return targets
    def construct_inputs(self):
        """x should have shape (batch, 1, numcells)

        :return: _description_
        """
        x = torch.tensor(self.x, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(self.y, dtype=torch.float32, requires_grad=True)
        inputs = torch.cat([x, y], dim=1).requires_grad_(True)
        return inputs, x, y

    def construct_targets(self):
        targets = torch.cat([self.u, self.v, self.p], dim=1)
        return targets

    def construct_cloud_inputs(self, k=2048):
        """x should have shape (batch, 1, numcells)

        :return: _description_
        """
        self.k = k

        if k == -1:
            x = torch.tensor(self.x, dtype=torch.float32, requires_grad=True)
            y = torch.tensor(self.y, dtype=torch.float32, requires_grad=True)
            inputs = torch.cat([x, y], dim=1).requires_grad_(True)
            return inputs, x, y, None
        else:
            perm = torch.randperm(self.x.shape[-1])
            # self.idx = perm[:k]
            indices = perm[:k]

            x = torch.tensor(
                self.x[:, :, indices], dtype=torch.float32, requires_grad=True
            )
            y = torch.tensor(
                self.y[:, :, indices], dtype=torch.float32, requires_grad=True
            )
            inputs = torch.cat([x, y], dim=1).requires_grad_(True)
            return inputs, x, y, indices

    def update_k(self, new_k=128):
        if not self._update_k:
            self._update_k = True
            self.k = new_k
            self.fixed_indices = torch.randperm(self.x.shape[-1])[: self.k]
        # else:
        #     print("not update k anymore")

    def construct_cloud_targets(self, indices=None):
        # assert self.idx.shape[-1] == self.k
        if indices == None:
            targets = torch.cat(
                [self.u, self.v, self.p],
                dim=1,
            )
            return targets
        else:

            targets = torch.cat(
                [
                    self.u[:, :, indices],
                    self.v[:, :, indices],
                    self.p[:, :, indices],
                ],
                dim=1,
            )
            return targets

    def construct_cloud_inputs_fix(self):
        """x should have shape (batch, 1, numcells)

        :return: _description_
        """

        x = torch.tensor(
            self.x[:, :, self.fixed_indices], dtype=torch.float32, requires_grad=True
        )
        y = torch.tensor(
            self.y[:, :, self.fixed_indices], dtype=torch.float32, requires_grad=True
        )
        inputs = torch.cat([x, y], dim=1).requires_grad_(True)
        return inputs, x, y, None

    def construct_cloud_targets_fix(self):
        targets = torch.cat(
            [
                self.u[:, :, self.fixed_indices],
                self.v[:, :, self.fixed_indices],
                self.p[:, :, self.fixed_indices],
            ],
            dim=1,
        )
        return targets
