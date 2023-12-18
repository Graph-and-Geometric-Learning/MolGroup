import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention


# this module is used for the meta gradient update in MolGroup
# if update if false, then the module is used for the vanilla forward pass
# if update is true, then the module is used for the meta gradient update
# for the vanilla gin, the update is always false
class ParamWrapper(nn.Module):
    def __init__(self):
        super(ParamWrapper, self).__init__()

    def forward(self, weight, update=False):
        if update and weight.grad is not None:
            return weight - weight.lr * weight.grad
        else:
            return weight


class BNLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.wrapper = ParamWrapper()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, update=False, bn_update=True):
        gamma = self.wrapper(self.gamma, update)
        beta = self.wrapper(self.beta, update)

        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            if bn_update:
                # auxiliary dataset should not update the running mean and var of target dataset
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = gamma * x + beta
        return x


class LNLayer(torch.nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LNLayer, self).__init__()
        self.wrapper = ParamWrapper()
        self.num_features = num_features
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x, update=False):
        gamma = self.wrapper(self.gamma, update)
        beta = self.wrapper(self.beta, update)
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = gamma * x + beta
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.wrapper = ParamWrapper()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, update=False):
        weight = self.wrapper(self.weight, update)
        bias = self.wrapper(self.bias, update)
        return torch.matmul(x, weight.t()) + bias


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.wrapper = ParamWrapper()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        
    def forward(self, x, update=False):
        emb = self.embedding(x)
        if update:
            # since the gradient is on the self.embedding.weight, we need to add the gradient on the extracted tensor
            if self.embedding.weight.grad is not None:
                emb.grad = self.embedding.weight.grad[x]
            emb.lr = self.embedding.weight.lr
        return self.wrapper(emb, update)


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([5, 6, 2]):
            emb = EmbeddingLayer(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.embedding.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr, update=False):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i], update)
        return bond_embedding


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        full_atom_feature_dims = [119, 4, 12, 12, 10, 6, 6, 2, 2]

        for i, dim in enumerate(full_atom_feature_dims):
            emb = EmbeddingLayer(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.embedding.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x, update=False, **kwargs):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i], update)

        return x_embedding


class NNDecoder(torch.nn.Module):
    def __init__(self, num_tasks_list, emb_dim = 300):
        super(NNDecoder, self).__init__()

        self.emb_dim = emb_dim
        self.num_tasks_list = num_tasks_list
        self.pool = global_mean_pool
        self.decoder = torch.nn.ModuleList([torch.nn.Linear(self.emb_dim, num_tasks) for num_tasks in num_tasks_list])

    def forward(self, batched_data, node_rep, task_idx=0):
        if len(node_rep.shape) == 3:
            # for graphormer
            h_graph = node_rep[:, 0, :]
        else:
            h_graph = self.pool(node_rep, batched_data.batch)
        return self.decoder[task_idx](h_graph)
