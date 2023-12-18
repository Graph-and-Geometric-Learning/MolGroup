import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem.Scaffolds import MurckoScaffold

from ogb.utils import smiles2graph
from ogb.graphproppred import PygGraphPropPredDataset

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data

from molgroup.utils.fingerprint_utils import smile2fp
from molgroup.utils.graphormer_collator import preprocess_item


def convert_to_pyg(graph_list):
    pyg_graph_list = []

    for graph in tqdm(graph_list):
        g = Data()
        g.num_nodes = graph['num_nodes']
        g.edge_index = torch.from_numpy(graph['edge_index'])

        del graph['num_nodes']
        del graph['edge_index']

        if graph['edge_feat'] is not None:
            g.edge_attr = torch.from_numpy(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = torch.from_numpy(graph['node_feat'])
            del graph['node_feat']

        pyg_graph_list.append(g)

    return pyg_graph_list


class MixedDataset(Dataset):
    def __init__(self, mixed_dataset_array, mixed_datasets=None, balance=False, iter_all_graphs=True):

        if not balance:
            self.mixed_dataset_array = torch.cat(mixed_dataset_array, dim=0)
        else:
            self.mixed_dataset_array = mixed_dataset_array

        self.mixed_datasets = mixed_datasets

        self.n_datasets = len(mixed_dataset_array)
        self.balance = balance
        if iter_all_graphs:
            self.num = sum([len(x) for x in mixed_dataset_array])
        else:
            assert self.balance
            # set the number of instances to be the number of graphs in the target dataset times the number of datasets
            # make sure that we can iterate through all the graphs in the target dataset
            self.num = len(mixed_dataset_array[0]) * self.n_datasets
        self.indices = torch.arange(self.num)

    def __getitem__(self, item):
        if not self.balance:
            # return the item-th instance
            sampled_idx = self.mixed_dataset_array[item]
        else:
            # balance sample. randomly pick a dataset and sample an instance from it
            dataset_id = torch.randint(0, self.n_datasets, (1,)).item()
            sampled_idx = self.mixed_dataset_array[dataset_id][torch.randint(0, len(self.mixed_dataset_array[dataset_id]), (1,)).item()]

        if self.mixed_datasets is not None:
            # directly return the sampled graph
            sampled_graph = self.mixed_datasets[sampled_idx[1]][sampled_idx[0]]
            return preprocess_item(sampled_graph), sampled_idx
        else:
            return sampled_idx

    def __len__(self):
        return self.num

    def shuffle(self):
        rand = torch.randperm(self.num)
        self.indices = self.indices[rand]


# this is a custom dataset that allows us to downsample/upsample the training set
class CustomPygDataset(PygGraphPropPredDataset):
    def __init__(self, name, root = 'dataset', n_train_graphs=5000, transform=None, pre_transform = None):
        self.n_train_graphs = n_train_graphs
        super(CustomPygDataset, self).__init__(name, root, transform, pre_transform)

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        
        custom_train_idx_path = osp.join(path, f'train_custom_{self.n_train_graphs}.pt')
        if os.path.exists(custom_train_idx_path):
            local_train_idx = torch.load(custom_train_idx_path)
        else:
            if len(train_idx) < self.n_train_graphs:
                # upsample
                local_train_idx = np.random.choice(len(train_idx), self.n_train_graphs).tolist()
            else:
                # downsample
                local_train_idx = np.random.choice(len(train_idx), self.n_train_graphs, replace=False).tolist()
            torch.save(local_train_idx, custom_train_idx_path)
        train_idx = train_idx[local_train_idx]

        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}


# for QM8 and QM9 with labels
class QMMolDataset(InMemoryDataset):
    def __init__(self, name, root, transform=None, pre_transform=None):
        self.name = name
        self.root = os.path.join(root, name)
        self.eval_metric = "mae"
        self.task_type = "regression"
        if self.name == "qm8":
            self.num_tasks = 16
        else:
            # qm9
            self.num_tasks = 19
        super(QMMolDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data_w_norm_labels.pt', 'scaffolds.pt'

    def get_idx_split(self, mode="random"):
        idx_path = os.path.join(self.root, f"{mode}_splits.pt")
        if not os.path.exists(idx_path):
            # random sample 3000 graphs for validation and test
            idx = torch.randperm(self.data.y.shape[0])
            train_idx = idx[:2000]
            valid_idx = idx[2000:2500]
            test_idx = idx[2500:3000]

            torch.save((train_idx, valid_idx, test_idx), idx_path)

        train_idx, valid_idx, test_idx = torch.load(idx_path)
        train_idx, valid_idx, test_idx = train_idx.long(), valid_idx.long(), test_idx.long()
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def process(self):
        raw_dataset = pd.read_csv(os.path.join(self.root, self.name + ".csv"))
        mols = [smiles2graph(smile) for smile in raw_dataset['smiles'].tolist()]
        scaffold_fps = [smile2fp(MurckoScaffold.MurckoScaffoldSmilesFromSmiles(x)) for x in raw_dataset['smiles'].tolist()]
        if self.name == "qm8":
            labels = raw_dataset.values[:, 1:]
        else:
            # qm9
            labels = raw_dataset.values[:, 2:]

        mols = convert_to_pyg(mols)

        # normalize the labels to be [0,1]
        labels = (labels - np.min(labels, axis=0)) / (np.max(labels, axis=0) - np.min(labels, axis=0))

        assert len(mols) == len(labels)
        new_mols, new_scaffolds = [], []
        for i, (g, label) in enumerate(zip(mols, labels)):
            scaffold_fp = scaffold_fps[i]
            if scaffold_fp is None or g is None or g.edge_index.shape[1] == 0:
                continue
            new_scaffolds.append(scaffold_fp)
            label = np.array(label, dtype=np.float32)
            g.y = torch.from_numpy(label).view(1,-1).to(torch.float32)
            new_mols.append(g)

        data, slices = self.collate(new_mols)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(new_scaffolds, self.processed_paths[1])
