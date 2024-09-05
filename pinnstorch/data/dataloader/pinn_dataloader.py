from typing import Any, Optional
import torch
import numpy as np
from pinnstorch.data.domains.point_cloud import PointData
from pinnstorch.data.domains.multi import CFDDomainSet
import glob
import os

# dataclass
# from pinnstorch.data import CFDDomain
from dataclasses import dataclass


@dataclass
class CFDDomain:
    all_collocation: PointData = None
    all_boundary: PointData = None
    wall: PointData = None
    obstacle: PointData = None
    inlet: PointData = None
    outlet: PointData = None


@dataclass
class CFDDomainWiseDataPoint:
    dd_collocation: list[PointData] = None  # len(dd_collocation) == batch_size
    dd_boundary: list[PointData] = None
    dd_wall: list[PointData] = None
    dd_obstacle: list[PointData] = None
    dd_inlet: list[PointData] = None
    dd_outlet: list[PointData] = None


class GeometryDataLoader_DELETED:
    """Custom DataLoader for Geometry Datasets

    This allow you to have a fast dataloader
    """

    def __init__(
        self,
        collocation_points: list[PointData] = None,
        boundary_points: list[PointData] = None,
        inlets: list[PointData] = None,
        outlets: list[PointData] = None,
        obstacles: list[PointData] = None,
        walls: list[PointData] = None,
        metadata: list[PointData] = None,
        domain_inputs: CFDDomainSet = None,
        batch_size: int = None,
        ignore: bool = False,
        shuffle: bool = False,
    ):
        """Initialize a GeometryDataLoader

        :param collocation_points: a list of PointData (collocation points) , with the length = number of domains
        :param boundary_points: a list of PointData (all boundary points)
        :param inlets: a list of PointData (only data points in inlets)
        :param outlets: a list of PointData (only data points in outlets)
        :param obstacles: a list of PointData (only data points in obstacles)
        :param walls: a list of PointData (only data points in obstacles)
        :param ignore: Whether to ignore incomplete batches (default is False).
        :param shuffle: Whether to shuffle the dataset (default is False)
        """
        super().__init__()

        self.collocation_points = np.array(collocation_points)
        self.boundary_points = np.array(boundary_points)

        self.inlets = np.array(inlets)
        self.outlets = np.array(outlets)
        self.obstacles = np.array(obstacles)
        self.walls = np.array(walls)
        self.metadata = np.array(metadata)
        # self.domains = domain_inputs

        self.batch_size = batch_size
        self.ignore = ignore
        self.shuffle = shuffle

        self.dataset_size = len(self.collocation_points)
        self.reset_indices()

    def reset_indices(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        else:
            self.indices = torch.arange(self.dataset_size)

    def __len__(self) -> int:
        """Get the number of batches in the dataloader

        :return: The number of batches
        """
        if self.batch_size is None:
            return 1

        if self.ignore:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size // self.dataset_size) + 1

    def __iter__(self):
        """Initialize the data loader iterator.

        :return: The data loader iterator
        """
        self.reset_indices()
        self.current_index = 0
        return self

    def __next__(self):
        """Generate the next batch of data

        :return: The next batch of data, together with metadata.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration

        # If batch_size is None, return the entire dataset as a single batch
        if self.batch_size is None:
            self.current_index += self.dataset_size
            raise NotImplementedError("batch_size=None")

        else:
            _indices = self.indices[
                self.current_index : self.current_index + self.batch_size
            ].tolist()
            coll = self.collocation_points[locations]
            bc = self.boundary_points[locations]
            inlet = self.inlets[locations]
            outlet = self.outlets[locations]
            obstacle = self.obstacles[locations]
            wall = self.walls[locations]
            self.current_index += self.batch_size
            return coll, bc, inlet, outlet, obstacle, wall


class GeometryDataLoader:
    """Custom DataLoader for Geometry Datasets

    This allow you to have a fast dataloader
    """

    def __init__(
        self,
        cfd_domains: list[CFDDomain] = None,
        batch_size: int = None,
        ignore: bool = False,
        shuffle: bool = False,
    ):
        """Initialize a GeometryDataLoader

        :param collocation_points: a list of PointData (collocation points) , with the length = number of domains
        :param boundary_points: a list of PointData (all boundary points)
        :param inlets: a list of PointData (only data points in inlets)
        :param outlets: a list of PointData (only data points in outlets)
        :param obstacles: a list of PointData (only data points in obstacles)
        :param walls: a list of PointData (only data points in obstacles)
        :param ignore: Whether to ignore incomplete batches (default is False).
        :param shuffle: Whether to shuffle the dataset (default is False)
        """
        super().__init__()

        self.cfd_domains = cfd_domains
        self.all_concatentated_matrix = {}
        self.all_concatentated_matrix["all_collocation_points"] = []
        self.all_concatentated_matrix["all_boundary_points"] = []
        self.all_concatentated_matrix["all_bc_wall"] = []
        self.all_concatentated_matrix["all_bc_inlet"] = []
        self.all_concatentated_matrix["all_bc_outlet"] = []
        self.all_concatentated_matrix["all_bc_obstacle"] = []

        for cfddomain in self.cfd_domains:
            self.all_concatentated_matrix["all_collocation_points"].append(
                cfddomain.all_collocation
            )
            self.all_concatentated_matrix["all_boundary_points"].append(
                cfddomain.all_boundary
            )
            self.all_concatentated_matrix["all_bc_wall"].append(cfddomain.wall)
            self.all_concatentated_matrix["all_bc_inlet"].append(cfddomain.inlet)
            self.all_concatentated_matrix["all_bc_outlet"].append(cfddomain.outlet)
            self.all_concatentated_matrix["all_bc_obstacle"].append(cfddomain.obstacle)

        for k in self.all_concatentated_matrix.keys():
            self.all_concatentated_matrix[k] = np.array(
                self.all_concatentated_matrix[k]
            )

        self.batch_size = batch_size
        self.ignore = ignore
        self.shuffle = shuffle

        self.dataset_size = len(self.cfd_domains)
        self.reset_indices()

    def reset_indices(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        else:
            self.indices = torch.arange(self.dataset_size)

    def __len__(self) -> int:
        """Get the number of batches in the dataloader

        :return: The number of batches
        """
        if self.batch_size is None:
            return 1

        if self.ignore:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size // self.batch_size) + 1

    def __iter__(self):
        """Initialize the data loader iterator.

        :return: The data loader iterator
        """
        self.reset_indices()
        self.current_index = 0
        return self

    def __next__(self) -> CFDDomainWiseDataPoint:
        """Generate the next batch of data

        :return: The next batch of data, together with metadata.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration

        # If batch_size is None, return the entire dataset as a single batch
        if self.batch_size is None:
            self.current_index += self.dataset_size
            raise NotImplementedError("self.batch_size is None")

        else:
            indices = self.indices[
                self.current_index : self.current_index + self.batch_size
            ].tolist()
            self.current_index += self.batch_size
            return CFDDomainWiseDataPoint(
                dd_collocation=self.all_concatentated_matrix["all_collocation_points"][
                    indices
                ],
                dd_boundary=self.all_concatentated_matrix["all_boundary_points"][
                    indices
                ],
                dd_inlet=self.all_concatentated_matrix["all_bc_inlet"][indices],
                dd_outlet=self.all_concatentated_matrix["all_bc_outlet"][indices],
                dd_obstacle=self.all_concatentated_matrix["all_bc_obstacle"][indices],
                dd_wall=self.all_concatentated_matrix["all_bc_wall"][indices],
            )


def to_point_data(pt_file, device="cpu") -> PointData:
    """Read .pth file data about a computional domain, and convert it to PointData type
    consisting of (x,y) coordinates, and physical fields (u, v, p)

    :param pt_file: path to the .pt (.pth) file
    :return: PointData, with 5 properties (x,y, u, v, p), each has a shape of (#batch_size, 1, #cell)
    """
    data = torch.load(pt_file)
    assert data.shape[0] == 5
    assert data.shape[1] > 100
    x = data[0, :].view(1, 1, -1).to(device)
    y = data[1, :].view(1, 1, -1).to(device)

    u = data[2, :].view(1, 1, -1).to(device)
    v = data[3, :].view(1, 1, -1).to(device)
    p = data[4, :].view(1, 1, -1).to(device)
    return PointData(x, y, u, v, p)


def load_data(data_dir, device="cpu") -> CFDDomainSet:
    domains = CFDDomainSet()
    for dirname in glob.glob(data_dir):
        for region in domains.regions:
            data = to_point_data(os.path.join(dirname, f"u+{region}.pt"), device)
            domains.add(data, region)
    return domains


def create_cfd_domain(data_dir, device="cpu") -> CFDDomain:
    """This function will read all .pt files in the `data_dir`, turn each file to PointData representation
    and gather all relevant PointData(s) to build a CFD Domain, which contain information about
    collocation/boundary points, points on wall/obstacle/inlet/out.

    :param data_dir: The path to the directory that contains .pt files
    :param device: device to store PointData, defaults to "cpu" (possible values: "cpu", "gpu")
    :return: CFDDomain dataclass
    """
    # domains = CFDDomainSet()

    return CFDDomain(
        all_collocation=to_point_data(
            os.path.join(data_dir, f"u+collocation_points.pt"), device
        ),
        all_boundary=to_point_data(
            os.path.join(data_dir, f"u+boundary_points.pt"), device
        ),
        wall=to_point_data(
            os.path.join(data_dir, f"u+boundary_points+wall.pt"), device
        ),
        obstacle=to_point_data(
            os.path.join(data_dir, f"u+boundary_points+obstacle.pt"), device
        ),
        inlet=to_point_data(
            os.path.join(data_dir, f"u+boundary_points+inlet.pt"), device
        ),
        outlet=to_point_data(
            os.path.join(data_dir, f"u+boundary_points+outlet.pt"), device
        ),
    )


def create_cfddomain_set(data_dirs, device="cpu") -> list[CFDDomain]:
    """This function will go through for directory in a [train/test] directory, and for each directory will use
    `create_cfd_domain` to construct `CFDDomain` obj.

    :param data_dirs: The path to the directory that contains cfd-domain/geometry sub-folder. (square_2, rectangle_1_1), etc.
    :param device: device to store PointData, defaults to "cpu" (possible values: "cpu", "gpu")
    :return: a list, each element represents a CFD domain (square_2) in terms of `CFDDomain`.
    """
    domain_set = []
    for cfd_dir in glob.glob(data_dirs):
        domain_set.append(create_cfd_domain(cfd_dir, device))
    return domain_set
