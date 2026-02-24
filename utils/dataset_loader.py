import inspect
import itertools
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    AddRandomWalkPE,
    BaseTransform,
    Compose,
    AddLaplacianEigenvectorPE
)
from torch_geometric.utils import degree
from torch_geometric.datasets import MoleculeNet

from utils.synthetic_triad_dataset import SyntheticTriadDataset
from utils.kHopSubgraphTransform import PrecomputeOneHopStructureOnly
from utils.kHopSubgraphTransform import custom_collate_fn

DATA_ROOT = os.getenv("DATA_ROOT", "./data")

import torch
import torch_geometric.transforms as T

regression_datasets = set(['alchemy_full'])


class PaddedLaplacianPE(T.AddLaplacianEigenvectorPE):
    def forward(self, data):
        # Save original k
        original_k = self.k
        
        # Safe k: at least 1, max num_nodes - 1
        k_safe = max(1, min(original_k, data.num_nodes - 1))
        
        # Temporarily set k_safe
        self.k = k_safe
        
        # Apply original transform
        data = super().forward(data)
        
        # Get pos_enc
        pe = getattr(data, self.attr_name)
        
        # PAD if needed
        if pe.size(1) < original_k:
            pad_size = original_k - pe.size(1)
            pad = torch.zeros(data.num_nodes, pad_size, device=pe.device, dtype=pe.dtype)
            pe = torch.cat([pe, pad], dim=1)
            setattr(data, self.attr_name, pe)
        
        # Restore original k (important!)
        self.k = original_k
        
        return data

    
def _get_labels(dataset):
    """Auxiliary function for returning labels of a data set."""
    labels = []

    for i in range(len(dataset)):
        labels.append(dataset[i].y)

    return labels

def _get_val2category_hiv():
    return {0: {
        1: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 11: 7, 12: 8, 14: 9,
        15: 10, 16: 11, 17: 12, 19: 13, 20: 14, 22: 15, 23: 16, 24: 17,
        25: 18, 26: 19, 27: 20, 28: 21, 29: 22, 30: 23, 31: 24, 32: 25,
        33: 26, 34: 27, 35: 28, 40: 29, 42: 30, 44: 31, 45: 32, 46: 33,
        47: 34, 50: 35, 51: 36, 52: 37, 53: 38, 55: 39, 64: 40, 65: 41,
        67: 42, 74: 43, 75: 44, 77: 45, 78: 46, 79: 47, 80: 48, 81: 49,
        82: 50, 83: 51, 89: 52, 92: 53
    }}

def get_num_feat_classes(dataset_name: str):
    if dataset_name == "HIV":
        return [54, 1, 11, 9, 5, 2, 7, 2, 2]
    

class ConvertToCategorical(BaseTransform):
    def __init__(self, mappings):
        self.mappings = mappings
    def __call__(self, data):
        for j in range(data["x"].shape[1]):
            if j not in self.mappings:
                continue
            val_to_idx = self.mappings[j]
            data["x"][:, j] = torch.tensor([val_to_idx[val.item()] for val in data["x"][:, j]])
        return data
    
class DropFeatureColumn(BaseTransform):
    def __init__(self, col_idx: int):
        self.col_idx = col_idx

    def __call__(self, data):
        # Ensure it's a 2D tensor
        if data['x'] is not None and data['x'].dim() == 2:
            mask = torch.ones(data.x.shape[1], dtype=torch.bool)
            mask[self.col_idx] = False
            data.x = data.x[:, mask]
        return data
    
def _get_class_ratios(dataset):
    """Auxiliary function for calculating the class ratios of a data set."""
    n_instances = len(dataset)

    labels = _get_labels(dataset)
    labels = [label.squeeze().tolist() for label in labels]
    ratios = np.bincount(labels).astype(float)
    ratios /= n_instances

    class_ratios = torch.tensor(ratios, dtype=torch.float32)
    return class_ratios


# def load_dataset():
#     dataset = TUDataset(root="data/letter-med", name="Letter-med")
#     dataset = dataset.shuffle()
#     train_size = int(len(dataset) * 0.8)
#     train_dataset = dataset[:train_size]
#     test_dataset = dataset[train_size:]
#     return train_dataset, test_dataset


class LapPETransform(BaseTransform):
    def __init__(self, k=10):
        self.k = k

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Convert edge index to adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1  # Undirected graph assumed

        # Compute Laplacian
        deg = torch.diag(adj.sum(dim=1))
        laplacian = deg - adj

        # Compute eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(
            laplacian
        )  # PyTorch version of eigen decomposition
        pe = eigvecs[:, 1 : self.k + 1]  # Exclude first eigenvector (trivial solution)

        # Fix sign flips
        pe = self.fix_sign_flips(pe, edge_index)

        # Store in data object
        data.pos_enc = pe
        return data

    def fix_sign_flips(self, pe, edge_index):
        """
        Aligns the sign of positional encodings using the highest-degree node.
        """
        num_nodes = pe.shape[0]

        # Compute node degrees
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)

        # Find the highest-degree node
        ref_node = torch.argmax(degrees).item()

        # Get the signs of the reference node's PE
        signs = torch.sign(pe[ref_node])

        # Flip signs consistently across all nodes
        return pe * signs


class RWPETransform(BaseTransform):
    def __init__(self, k=10):
        self.k = k
        super().__init__()

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Convert edge index to adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1

        # Compute degree matrix
        deg = torch.diag(adj.sum(dim=1))

        # Compute transition matrix
        deg_inv = torch.diag(1.0 / (deg.diag() + 1e-6))
        P = torch.matmul(deg_inv, adj)

        # Compute RWPE
        rwpe = [P]  # Start with 1-step probabilities
        for _ in range(1, self.k):
            rwpe.append(torch.matmul(rwpe[-1], P))  # Multiply by P to simulate walks

        rwpe = torch.stack(rwpe, dim=-1)  # Shape: (num_nodes, num_nodes, k)
        rwpe = rwpe.sum(dim=1)  # Sum over columns to get per-node features

        data.pos_enc = rwpe  # Store RWPE in the data object
        return data


class MyTransform(BaseTransform):
    def __init__(self, k=2):
        self.k = k
        super().__init__()

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # adjaceny matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1

        # laplacian matrix
        deg = torch.diag(adj.sum(dim=1))
        laplacian = deg - adj

        _, eigvecs = torch.linalg.eigh(laplacian)

        if eigvecs.shape[1] < self.k + 1:
            pe = torch.zeros(
                (num_nodes, self.k)
            )  # Zero padding if there aren't enough eigenvectors
            pe = torch.zeros((num_nodes, self.k))  # Create a zero matrix
            existing_k = (
                eigvecs.shape[1] - 1
            )  # Number of available eigenvectors (excluding the first one)
            pe[:, :existing_k] = eigvecs[
                :, 1 : existing_k + 1
            ]  # Copy existing eigenvectors
        else:
            pe = eigvecs[
                :, 1 : self.k + 1
            ]  # Exclude first eigenvector (trivial solution)

        data.pos_enc = pe
        # print(f'num_nodes: {num_nodes}')
        # print(f'edge_index shape: {edge_index.shape}')
        return data
    
class OneHotDecoding(BaseTransform):

    def __call__(self, data):
        """Adjust multi-class labels (reverse one-hot encoding).

        This is necessary because some data sets use one-hot encoding
        for their labels, wreaks havoc with some multi-class tasks.
        """
        label = data["y"]

        if len(label.shape) > 1:
            label = label.squeeze().tolist()

            if isinstance(label, list):
                label = label.index(1.0)

            data["y"] = torch.as_tensor([label], dtype=torch.long)

        return data


class SmallGraphDataset(pl.LightningDataModule):

    def __init__(
        self,
        name,
        batch_size,
        val_fraction=0.1,
        test_fraction=0.1,
        fold=0,
        seed=42,
        n_splits=5,
        pe_name ='rwpe',
        pos_enc_dim = 10,
        num_hops = [1]
    ):
        super().__init__()
        self.is_regression = name in regression_datasets
        self.name = name
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        self.n_splits = n_splits
        self.fold = fold
        pre_transform_list = [PrecomputeOneHopStructureOnly(num_hops=num_hops)] # Common pre_transform for all datasets
        transform_list =[]
        
        
        if pe_name == 'rwpe':
            transform_list.append(AddRandomWalkPE(walk_length=pos_enc_dim, attr_name="pos_enc"),)
            print('using rwpe')

        elif pe_name == 'lape':
            print('using lape')
            transform_list.append(PaddedLaplacianPE(k=pos_enc_dim, attr_name='pos_enc', is_undirected=True))
        
        else:
            raise NotImplementedError(f'The positional encoding {pe_name} is not implemented')

        if self.name == "HIV":
            self.base_class = MoleculeNet
            self.root = os.path.join(DATA_ROOT, "ModelNet")
            pre_transform_list.append(ConvertToCategorical(_get_val2category_hiv()))
            transform_list.append(OneHotDecoding())
        elif self.name == "SYN":
            pre_transform_list = []
            transform_list = []
            self.root = os.path.join(DATA_ROOT, "SYN")
            self.base_class = SyntheticTriadDataset
        else:
            self.base_class = TUDataset
            self.root = os.path.join(DATA_ROOT, "TU")


        self.pre_transform = Compose(pre_transform_list)
        self.transform = Compose(transform_list)

    def _get_max_degree(self):
        """Auxiliary function for getting the maximum degree of data set."""
        dataset = self.base_class(root=self.root, name=self.name)

        max_degrees = torch.as_tensor(
            [
                torch.max(
                    degree(data.edge_index[0, :], data.num_nodes, dtype=torch.int)
                )
                for data in dataset
            ]
        )

        return torch.max(max_degrees)

    def __prepare_data(self):
        args = {
            "root": self.root,
            "name": self.name,
            "transform": self.transform,
            "pre_transform": self.pre_transform,
        }

        if "use_node_attr" in inspect.signature(self.base_class.__init__).parameters:
            args["use_node_attr"] = True

        dataset = self.base_class(**args)

        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features
        if self.name == "HIV":
            self.num_features = 44

        n_instances = len(dataset)
        labels = _get_labels(dataset) if not self.is_regression else None
        self.class_ratios = None if self.is_regression else _get_class_ratios(dataset)

        if self.is_regression:
            splitter = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
            split_iterator = splitter.split(torch.arange(n_instances))
        else:
            splitter = StratifiedKFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
            split_iterator = splitter.split(torch.arange(n_instances), torch.tensor(labels))

        train_index, test_index = next(itertools.islice(split_iterator, self.fold, None))
        train_index, val_index = train_test_split(train_index, random_state=self.seed)

        train_index = train_index.tolist()
        val_index = val_index.tolist()
        test_index = test_index.tolist()

        if self.is_regression:
            for i in train_index:
                dataset._data_list[i].y = torch.zeros_like(dataset[i].y)


        self.train = Subset(dataset, train_index)
        self.val = Subset(dataset, val_index)
        self.test = Subset(dataset, test_index)
        ys = torch.vstack([dataset[i].y for i in train_index])
        print("Train targets mean:", ys.mean(dim=0))
        print("Train targets std:", ys.std(dim=0))


    def prepare_data(self, force_reload = False):
        # Prepare parameters for the base class. This is somewhat
        # tedious because the `use_node_attr` is not available in
        # all base classes.
        args = {
            "root": self.root,
            "name": self.name,
            "transform": self.transform,
            "pre_transform": self.pre_transform,
        }

        if "use_node_attr" in inspect.signature(self.base_class.__init__).parameters:
            args["use_node_attr"] = True

        dataset = self.base_class(**args, force_reload=force_reload)

        self.num_classes = dataset.num_classes
        print(self.num_classes)
        self.num_features = dataset.num_features
        if self.name == "HIV":
            self.num_features = 44
            
        n_instances = len(dataset)
        labels = _get_labels(dataset) if not self.is_regression else None

        self.class_ratios = None if self.is_regression else _get_class_ratios(dataset)

        # if self.is_regression:
        #     skf = StratifiedKFold(
        #         n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        # else:
        #     skf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)

        # skf_iterator = skf.split(
        #     torch.tensor([i for i in range(n_instances)]),
        #     torch.tensor(labels),
        # )
        if self.is_regression:
            splitter = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
            split_iterator = splitter.split(torch.arange(n_instances))
        else:
            splitter = StratifiedKFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
            split_iterator = splitter.split(torch.arange(n_instances), torch.tensor(labels))

        #train_index, test_index = next(itertools.islice(skf_iterator, self.fold, None))
        train_index, test_index = next(itertools.islice(split_iterator, self.fold, None))

        train_index, val_index = train_test_split(train_index, random_state=self.seed)

        train_index = train_index.tolist()
        val_index = val_index.tolist()
        test_index = test_index.tolist()

        if self.is_regression:

            self.y_scaler = StandardScaler()
            # Extract and reshape y
            y_train = torch.vstack([dataset[i].y.reshape(1, -1) for i in train_index]).numpy()
            print(y_train.shape)
            self.y_scaler.fit(y_train)

            for i in train_index:
                y = dataset[i].y.view(1, -1).numpy()
                y_scaled = self.y_scaler.transform(y)
                dataset._data_list[i].y = torch.tensor(y_scaled, dtype=torch.float32)

            for i in val_index:
                y = dataset[i].y.view(1, -1).numpy()
                y_scaled = self.y_scaler.transform(y)
                dataset._data_list[i].y = torch.tensor(y_scaled, dtype=torch.float32)

            for i in test_index:
                y = dataset[i].y.view(1, -1).numpy()
                y_scaled = self.y_scaler.transform(y)
                dataset._data_list[i].y = torch.tensor(y_scaled, dtype=torch.float32)

        self.train = Subset(dataset, train_index)
        self.val = Subset(dataset, val_index)
        self.test = Subset(dataset, test_index)

    def train_dataloader(self):
        print('[train_dataloader]')
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
