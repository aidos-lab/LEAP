import json
import os
from enum import Enum

import dect
import dect.ect
import pandas as pd
import torch
import torch_geometric.nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph, scatter


class ECTtype(Enum):
    POINTS = "points"
    EDGES = "edges"


class ECTembedMethod(Enum):
    LINEAR = "linear"
    ATTN = "attn"
    ATTN_PE = "attn_pe"
    DEEPSETS = "deepsets"
    CONV1D = "conv1d"


class ConvType(Enum):
    GCNConv = "GCNConv"
    GATConv = "GATConv"

    @staticmethod
    def convert(name):
        if name == ConvType.GCNConv.value:
            return torch_geometric.nn.GCNConv
        if name == ConvType.GATConv.value:
            return torch_geometric.nn.GATConv

        raise NotImplementedError(f"{name} conv type is not available")


def params2dict(**kwargs):
    """
    Convert a list of parameters to a dictionary.
    """
    return kwargs


def save_dict_as_json(myDict: dict, filename: str):
    with open(filename, "w") as file:
        json.dump(myDict, file, indent=4)


def load_dict_from_json(filename: str):
    with open(filename, "r") as file:
        myDict = json.load(file)
    return myDict


def print_dict(myDict: dict):
    for key, value in myDict.items():
        print(f"{key}: {value}")


def dict2csv(data_dict, file_path):
    df = pd.DataFrame([data_dict])  # Create a one-row DataFrame
    file_exists = os.path.exists(file_path)
    write_header = not file_exists or os.stat(file_path).st_size == 0
    df.to_csv(file_path, mode="a", header=write_header, index=False)


def maxL2_graph_norm(x, batch, eps=1e-6):
    # x: shape (num nodes, num features) - node features
    # batch: shape (num nodes) - graph assignment for each node

    # compute features means per graph
    mean = scatter(x, batch, dim=0, reduce="mean")  # shape (num graphs, num features)
    x_centered = x - mean[batch]  # broadcasting of mean to be (num nodes, num features)

    # Compute L2 norm
    l2_norm = x_centered.norm(p=2, dim=1, keepdim=True)

    # compute max L2 norm per graph
    max_l2 = scatter(l2_norm, batch, dim=0, reduce="max")
    max_l2 = max_l2[batch] + eps

    # normalize
    x_normed = x_centered / max_l2

    return x_normed


def build_neigh_batch(x, e_index, ptr, subgraph_info=None, hops=1):
    all_subgraphs = []
    if subgraph_info:
        for graph_idx, subgraph_info_list in enumerate(subgraph_info[hops]):
            node_offset = ptr[graph_idx]
            # Pre-batch subsets and edge_index tensors
            for info in subgraph_info_list:
                subset = info["subset"]
                edge_index = info["edge_index"]
                global_subset = subset + node_offset
                all_subgraphs.append(Data(x=x[global_subset], edge_index=edge_index))
    else:
        for i in range(x.shape[0]):
            subset, edge_index, _, _ = k_hop_subgraph(
                node_idx=i,
                num_hops=1,
                edge_index=e_index,
                relabel_nodes=True,
                num_nodes=x.shape[0],
            )
            all_subgraphs.append(Data(x=x[subset], edge_index=edge_index))

    return Batch.from_data_list(all_subgraphs)


def ect_wrapper(
    x: torch.Tensor,
    v: torch.Tensor,
    radius: float,
    resolution: int,
    scale: torch.Tensor,
    normalize: bool,
    type: str = "edges",
    edge_index: torch.Tensor = None,
) -> torch.Tensor:
    """
    Wrapper for the ECT computation.
    """
    if type == "edges":
        ect = dect.ect.compute_ect_edges(
            x=x,
            edge_index=edge_index,
            v=v,
            radius=radius,
            resolution=resolution,
            scale=scale,
        )
    elif type == "points":
        ect = dect.ect.compute_ect_point_cloud(
            x=x.unsqueeze(0),
            v=v,
            radius=radius,
            resolution=resolution,
            scale=scale,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown ECT type: {type}")

    return ect


def compute_local_ect(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    v: torch.Tensor,
    radius: float = 1.1,
    resolution: int = 16,
    scale: int = 500,
    normalize_after: bool = False,
    normalize_before: bool = True,
    hops: int = 1,
    type: str = "edges",
) -> torch.Tensor:
    """
    Compute the local ECT in the subgraph.
    """
    scale = torch.tensor([scale]).to(x.device)
    l_ects = torch.zeros(x.shape[0], resolution * v.shape[1]).to(x.device)

    # for each node in the graph compute the ECT in a neighborhood
    for i in range(x.shape[0]):
        # compute neighborhood

        subset, e_index, _, _ = k_hop_subgraph(
            node_idx=i,
            num_hops=1,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=x.shape[0],
        )

        if normalize_before:
            # center around 0 and inside the unit sphere
            neighbors_features = x[subset] - torch.mean(x[subset], dim=0, keepdim=True)
            max_norm = torch.norm(neighbors_features, dim=1).max()
            neighbors_features = neighbors_features / (max_norm + 1e-6)

        # compute ECT in the subgraph (l-ECT)
        lect = ect_wrapper(
            x=neighbors_features,
            v=v,
            radius=radius,
            resolution=resolution,
            scale=scale,
            normalize=normalize_after,
            type=type,
            edge_index=e_index,
        )
        lect = torch.flatten(lect)
        l_ects[i] = lect

    return l_ects
