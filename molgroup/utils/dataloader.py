from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from molgroup.utils.dataset import MixedDataset
from molgroup.utils.graphormer_collator import collator


class GraphormerMixedDataLoader(DataLoader):
    def __init__(self, dataset_list, fp_list=None, balance=False, iter_all_graphs=True, **kwargs):
        
        n_graphs = [len(dataset) for dataset in dataset_list]
        self.n_dataset = len(dataset_list)
        self.datasets = dataset_list
        
        self.fp_list = fp_list
        if fp_list is not None:
            for i in range(self.n_dataset):
                self.datasets[i].data.ft_feat = self.fp_list[i]

        mixed_instances_array = []
        for i, n_graph in enumerate(n_graphs):
            mixed_instances_array.append(torch.cat([torch.arange(n_graph).unsqueeze(dim=1), 
                        torch.ones(n_graph).unsqueeze(dim=1)*i], dim=1).long())

        self.dataset = MixedDataset(mixed_instances_array, mixed_datasets=dataset_list, 
                                    balance=balance, iter_all_graphs=iter_all_graphs)

        self.collator = partial(collator, max_node=128, multi_hop_max_dist=5, rel_pos_max=1024)

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=self.dataset, **kwargs)

    def __collate_fn__(self, batch):
        batch_graphs = [x[0] for x in batch]
        batch_idx = torch.stack([x[1] for x in batch])
        ret_batch = []
        
        for i in range(self.n_dataset):
            mask = batch_idx[:, 1] == i
            if mask.sum() > 0:
                idx = torch.where(mask)[0]
                sampled_graphs = [batch_graphs[i] for i in idx]
                sampled_batch = self.collator(sampled_graphs)  # make the sampled graphs a batch

                if sampled_batch is not None:
                    sampled_batch.dataset_id = torch.ones(sampled_batch.x.shape[0], dtype=torch.long) * i

                ret_batch.append(sampled_batch)
            else:
                ret_batch.append(None)
        return ret_batch


class DataLoaderFP(DataLoader):
    def __init__(self, mol_dataset, fp_feat=None, dataset_id=0, **kwargs):
        
        self.mol_dataset = mol_dataset
        self.dataset_id = dataset_id

        assert fp_feat is None or len(mol_dataset) == len(fp_feat)
        self.fp_feat = fp_feat

        array = torch.arange(len(mol_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)
    
    def __collate_fn__(self, batch_idx):
        batch_idx = torch.stack(batch_idx, dim=0)
        sampled_batch = Batch.from_data_list(self.mol_dataset[batch_idx], None, None)

        if self.fp_feat is not None:
            ptr = sampled_batch.ptr
            index_list = [torch.tensor([feat_id] * (e - s)) for feat_id, (s, e) in enumerate(zip(ptr[:-1], ptr[1:]))]
            index_list = torch.cat(index_list, dim=0)
            sampled_batch.fp_feat = self.fp_feat[batch_idx][index_list]

        sampled_batch.dataset_id = torch.ones(sampled_batch.x.shape[0], dtype=torch.long) * self.dataset_id
        
        return sampled_batch


class MixedDataLoader(DataLoader):
    def __init__(self, dataset_list, fp_list=None, balance=False, iter_all_graphs=True, **kwargs):
        
        n_graphs = [len(dataset) for dataset in dataset_list]
        self.n_dataset = len(dataset_list)
        self.datasets = dataset_list
        
        self.fp_list = fp_list
        if fp_list is not None:
            for i in range(self.n_dataset):
                self.datasets[i].data.ft_feat = self.fp_list[i]

        mixed_instances_array = []
        for i, n_graph in enumerate(n_graphs):
            mixed_instances_array.append(torch.cat([torch.arange(n_graph).unsqueeze(dim=1), 
                        torch.ones(n_graph).unsqueeze(dim=1)*i], dim=1).long())

        self.dataset = MixedDataset(mixed_instances_array, balance=balance, iter_all_graphs=iter_all_graphs)

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=self.dataset, **kwargs)

    def __collate_fn__(self, batch_idx):
        batch_idx = torch.stack(batch_idx, dim=0)        
        ret_batch = []

        for i in range(self.n_dataset):
            mask = batch_idx[:, 1] == i
            if mask.sum() > 0:
                idx = batch_idx[mask, 0].view(-1)
                sampled_batch = Batch.from_data_list(self.datasets[i][idx], None, None)

                if self.fp_list is not None:
                    # use ptr to assign fp_feat to each node in the batch        
                    ptr = sampled_batch.ptr
                    index_list = [torch.tensor([feat_id] * (e - s)) for feat_id, (s, e) in enumerate(zip(ptr[:-1], ptr[1:]))]
                    index_list = torch.cat(index_list, dim=0)
                    sampled_batch.fp_feat = self.fp_list[i][idx][index_list]            
                    # # assign fp_feat to each instance
                    sampled_batch.mol_fp_feat = self.fp_list[i][idx]
                else:
                    sampled_batch.fp_feat = None

                sampled_batch.dataset_id = torch.ones(sampled_batch.x.shape[0], dtype=torch.long) * i

                ret_batch.append(sampled_batch)
            else:
                ret_batch.append(None)
        return ret_batch
