import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import Set2Set


class GaterWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_tasks, gate_temp=0.1, mix_alpha=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.gaters = MixGate(num_tasks, input_size, hidden_size, gate_temp, mix_alpha)
        
    def forward(self, x, task_id, **kwargs):
        return self.gaters(kwargs["target_x"], x, kwargs["target_batch_data"], kwargs["aux_batch_data"], task_id)
        
    def gating_score(self, x=None):
        return self.gaters.all_gating_score()


class MixGate(nn.Module):
    def __init__(self, n_tasks, input_size, hidden_size, gate_temp=0.1, alpha=0.1):
        super().__init__()

        self.alpha = alpha
        self.n_tasks = n_tasks
        self.temp = gate_temp
        self.task_embedding = torch.rand(n_tasks, hidden_size)
        torch.nn.init.orthogonal_(self.task_embedding)
        if self.n_tasks > 1:
            # let all the auxiliary tasks have the same initialization as the first auxiliary task
            self.task_embedding[1:] = self.task_embedding[1]
        self.task_embedding = torch.nn.Parameter(self.task_embedding)

        self.structure_embeddings = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.graph_pool = global_mean_pool  
        self.structure_pool = nn.ModuleList([Set2Set(hidden_size, processing_steps = 1) for _ in range(n_tasks)])
        self.norm = torch.nn.functional.normalize

        # let gate score be zero at the beginning of training
        self.ini_structure_embeddings = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.ini_structure_pool = nn.ModuleList([Set2Set(hidden_size, processing_steps = 2) for _ in range(n_tasks)])
        self.ini_structure_embeddings.load_state_dict(self.structure_embeddings.state_dict())
        self.ini_structure_pool.load_state_dict(self.structure_pool.state_dict())

    def forward(self, target_x, aux_x, target_batch, aux_batch, task_id, return_all=True):

        task_embedding = self.norm(self.task_embedding, dim=-1)

        target_embedding = self.norm(self.graph_pool(self.structure_embeddings(target_x), target_batch), dim=-1)
        aux_embedding = self.norm(self.graph_pool(self.structure_embeddings(aux_x), aux_batch), dim=-1)
        target_structure_emb = self.norm(self.structure_pool[0](target_embedding), dim=-1)
        aux_structure_emb = self.norm(self.structure_pool[task_id](aux_embedding), dim=-1)

        structure_affinity = (target_structure_emb * aux_structure_emb).sum(dim=-1)
        task_affinity = (task_embedding[0] * task_embedding[task_id]).sum(dim=-1)

        if return_all:
            return torch.sigmoid(((1 - self.alpha) * task_affinity + self.alpha * structure_affinity) / self.temp), torch.sigmoid(task_affinity), torch.sigmoid(structure_affinity)
        return torch.sigmoid(((1 - self.alpha) * task_affinity + self.alpha * structure_affinity) / self.temp)

    def gating_score(self, task_id):
        task_embedding = self.norm(self.task_embedding, dim=-1)
        return torch.sigmoid((task_embedding[0] * task_embedding[task_id]).sum(dim=-1) / self.temp).cpu()

    def all_gating_score(self):
        task_embedding = self.norm(self.task_embedding, dim=-1)
        return torch.stack([torch.sigmoid((task_embedding[0] * task_embedding[task_id]).sum(dim=-1) / self.temp).cpu() \
                            for task_id in range(self.n_tasks)], dim=0)
