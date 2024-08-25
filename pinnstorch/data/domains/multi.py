from pinnstorch.data.domains.point_cloud import PointData
import numpy as np


class CFDDomainSet:
    def __init__(self, domains=None):
        self.regions = ["collocation_points", "boundary_points"]
        self.regions += [
            "boundary_points+" + patch
            for patch in ["inlet", "outlet", "obstacle", "wall"]
        ]
        if not domains:
            self.domains = {k: list() for k in self.regions}  # list(PointData)
        else:
            self.domains = domains
            # convert to numpy array
            for r in self.regions:
                if not isinstance(self.domains[r], np.ndarray):
                    self.domains[r] = np.array(self.domains[r])

    def add(self, data: PointData, key="boundary_points"):
        assert key in self.regions
        self.domains[key].append(data)
        if not isinstance(self.domains[key], np.ndarray):
            self.domains[key] = np.array(self.domains[key])

    # def sample(self, indices):
    #     data = {}  # {k: list() for k in self.regions}
    #     for r in self.regions:
    #         data[r] = self.domains[r][indices]
    #     return data
    #     # CFDDomainSet(domains=data)

    # @property
    # def loop_generator(self):
    #     for i in range(len(self.domains["collocation_points"])):
    #         yield [self.domains[r][i] for r in self.regions]
