import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from molgroup.models.gater import GaterWrapper
from molgroup.models.model_utils import ParamWrapper, BNLayer, LinearLayer, AtomEncoder, BondEncoder


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


class MetaAtomEncoderGate(torch.nn.Module):
    def __init__(self, emb_dim, n_tasks):
        super(MetaAtomEncoderGate, self).__init__()

        self.n_tasks = n_tasks
        self.emb_dim = emb_dim
        self.atom_encoders = torch.nn.ModuleList([AtomEncoder(emb_dim) for _ in range(n_tasks)])

    def ini_same_weight(self):
        for i in range(1, len(self.atom_encoders)):
            for name, param in self.atom_encoders[i].named_parameters():
                param.data = self.atom_encoders[0].state_dict()[name]

    def forward(self, x, dataset_idx, gate=None, update=False):
        res = self.atom_encoders[dataset_idx](x, update)
        if dataset_idx != 0 and gate is not None:
            return gate * res + (1 - gate) * self.atom_encoders[0](x, update)
        else:
            return res


class MetaGINConv(MessagePassing):
    # target dataset only uses the shared parameters
    # other datasets use the combination of shared and dataset-specific parameters
    # shared/specific bondencoder
    def __init__(self, emb_dim, n_datasets):
        super(MetaGINConv, self).__init__(aggr = "add")

        self.n_datasets = n_datasets
        self.mlps = torch.nn.ModuleList([TwoLayerMLP(emb_dim) for _ in range(n_datasets)])
        self.eps = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor([0])) for _ in range(n_datasets)])
        self.bond_encoders = torch.nn.ModuleList([BondEncoder(emb_dim = emb_dim) for _ in range(n_datasets)])
        self.wrapper = ParamWrapper()

    def ini_same_weight(self):
        for i in range(1, len(self.mlps)):
            for name, param in self.mlps[i].named_parameters():
                param.data = self.mlps[0].state_dict()[name]
        for i in range(1, len(self.bond_encoders)):
            for name, param in self.bond_encoders[i].named_parameters():
                param.data = self.bond_encoders[0].state_dict()[name]                

    def forward(self, x, edge_index, edge_attr, dataset_idx, gate=None, update=False):
        edge_embedding = self.bond_encoders[dataset_idx](edge_attr, update)            
        eps = self.wrapper(self.eps[dataset_idx], update)
        out = self.mlps[dataset_idx]((1 + eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding), update=update)

        if dataset_idx != 0 and gate is not None:
            target_edge_embedding = self.bond_encoders[0](edge_attr, update)
            target_eps = self.wrapper(self.eps[0], update)
            target_out = self.mlps[0]((1 + target_eps) * x + self.propagate(edge_index, x=x, edge_attr=target_edge_embedding), 
                                        update=update, bn_update=False)
            return gate * out + (1 - gate) * target_out 
        else:
            return out

    def get_device(self):
        return next(self.parameters()).device

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    # BN module can only be updated by target dataset
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = TwoLayerMLP(emb_dim)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, dataset_idx, **kwargs):
        device = self.get_device()
        edge_embedding = self.bond_encoder(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, 
                                                edge_attr=edge_embedding), 
                                                bn_update=dataset_idx==0)

    def get_device(self):
        return next(self.parameters()).device

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class MetaGIN(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, n_datasets, 
                gate_input_dim=None, gate_hidden_dim=None,
                drop_ratio = 0.5, gate_temp = 1.0, gate_mix_alpha = 0.1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(MetaGIN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.n_datasets = n_datasets

        self.gater = GaterWrapper(gate_input_dim, gate_hidden_dim, num_tasks=self.n_datasets, gate_temp=gate_temp, mix_alpha=gate_mix_alpha)

        self.atom_encoder = MetaAtomEncoderGate(emb_dim, n_datasets)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.convs.append(MetaGINConv(emb_dim, n_datasets))

    def _get_gater(self, input_dim, emb_dim, gate_temp, gate_mix_alpha):
        return GaterWrapper(input_dim, emb_dim, num_tasks=self.n_datasets, gate_temp=gate_temp, mix_alpha=gate_mix_alpha)

    def gating_score(self):
        gates = []
        gates.append(self.gater.gating_score().view(-1))
        gates = torch.stack(gates, dim=0)#.cpu().tolist()
        return gates

    def ini_same_weight(self):
        assert self.gate_model is not None
        self.atom_encoder.ini_same_weight()
        for layer in range(self.num_layer):
            self.convs[layer].ini_same_weight()

    def forward(self, batched_data, dataset_idx=0, return_gate_score = False, update=False, use_gate=True, **kwargs):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        try:
            fp_feat = batched_data.fp_feat
        except:
            fp_feat = None

        if use_gate and dataset_idx != 0:
            gate_scores, task_affinity, structure_affinity = self.gater(fp_feat, dataset_idx, aux_batch_data=batch, **kwargs)
            gate_scores = gate_scores.to(x.device)
            gate_scores_ret = [gate_scores, task_affinity, structure_affinity]
        else:
            gate_scores_ret = gate_scores = None

        ### computing input node embedding
        h_list = [self.atom_encoder(x, dataset_idx=dataset_idx, gate=gate_scores, update=update)]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr, 
                           dataset_idx, gate=gate_scores, update=update)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)

        node_representation = h_list[-1]
        if return_gate_score:
            return node_representation, gate_scores_ret
        else:
            return node_representation

    def get_device(self):
        return next(self.parameters()).device
