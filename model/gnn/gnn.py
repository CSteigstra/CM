from torch import nn
# from torch_geometric.data import Data
import torch_geometric.nn as geom_nn
import torch

def build_indexing(x):
    # res = []
    *_, h, w, dim = x.shape
    # print(x.shape)
    device = x.device

    # i = torch.arange(h)
    # j = torch.arange(w)

    i, j = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    i, j = i.flatten(), j.flatten()

    res = torch.cat((
        ((i - 1) % h) * h + (j - 1) % w,
        ((i - 1) % h) * h + j % w,
        ((i - 1) % h) * h + (j + 1) % w,
        (i % h) * h + (j - 1) % w,
        (i % h) * h + j % w,
        (i % h) * h + (j + 1) % w,
        ((i - 1) % h) * h + (j - 1) % w,
        ((i - 1) % h) * h + j % w,
        ((i - 1) % h) * h + (j + 1) % w,
    ))

    res = torch.stack((res,
        (i[:, None].repeat(1, 9) * 4 + j[:, None].repeat(1, 9)).flatten()
    ))

    # for i in range(h):
    #     for j in range(w):
    #         res.append(((i - 1) % h) * h + (j - 1) % w)
    #         res.append(((i - 1) % h) * h + j % w)
    #         res.append(((i - 1) % h) * h + (j + 1) % w)

    #         res.append((i % h) * h + (j - 1) % w)
    #         res.append((i % h) * h + j % w)
    #         res.append((i % h) * h + (j + 1) % w)

    #         res.append(((i + 1) % h) * h + (j - 1) % w)
    #         res.append(((i + 1) % h) * h + j % w)
    #         res.append(((i + 1) % h) * h + (j + 1) % w)

    return res

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = geom_nn.GCNConv
        # self.edge_index = build_indexing()
        # gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        
        edge_index = build_indexing(x)[None, :].repeat(x.shape[0], 1, 1)
        x = x.view(x.shape[0], -1, 4)
        
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x