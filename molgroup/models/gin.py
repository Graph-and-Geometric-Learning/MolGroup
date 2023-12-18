import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from molgroup.models.model_utils import BNLayer, LinearLayer, AtomEncoder, BondEncoder


class TwoLayerMLP(nn.Module):
    def __init__(self, dim):
        super(TwoLayerMLP, self).__init__()
        self.weight = LinearLayer(dim, 2*dim)
        self.act = nn.ReLU()
        self.bn = BNLayer(2*dim)
        self.weight2 = LinearLayer(2*dim, dim)
        self.bn2 = BNLayer(dim)

    def forward(self, x, update=False, bn_update=True):
        x = self.bn(self.weight(x, update), update, bn_update)
        x = self.act(x)
        return self.bn2(self.weight2(x, update), update, bn_update)


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = TwoLayerMLP(emb_dim)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, dataset_idx):
        edge_embedding = self.bond_encoder(edge_attr)
        # BN module can only be updated by target dataset
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, 
                                                edge_attr=edge_embedding), 
                                                bn_update=dataset_idx==0)

    def get_device(self):
        return next(self.parameters()).device

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GIN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5):
        super(GIN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim))

    def forward(self, batched_data, dataset_idx=0):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr, dataset_idx)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)

        return h_list[-1]

    def get_device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    model = GIN(num_layer=5, emb_dim=300, n_datasets=1)