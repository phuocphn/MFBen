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
            print(f"Initialized weight for: {m}")

    def forward(self, x):
        """Pass inputs to the self.net to obtain the outputs

        :param x: Input tensor
        :return: Output tensor, with the shape (num_cells, self.num_outputs)
        """
        x = self.net(x)
        return x
