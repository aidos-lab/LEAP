import os, math, random
from typing import List, Optional, Union

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def _sample_unit_disk_point():
    r = math.sqrt(random.random())
    theta = 2.0 * math.pi * random.random()
    return (r * math.cos(theta), r * math.sin(theta))


class SyntheticTriadDataset(InMemoryDataset):
    """
    Build 4*N graphs of 3 nodes each from the same 3 sampled 2D points (||x||<=1):
      y=0: no edges
      y=1: 1 random edge
      y=2: 2 random edges
      y=3: 3 edges (triangle)
    - Undirected graphs
    - NO self-loops
    - No transforms are needed; __getitem__ returns raw items (ignores transform)
    """

    def __init__(self,
                 root: str,
                 name: Optional[str] = None,       # accepted for compat; ignored
                 transform=None,
                 pre_transform=None,
                 force_reload: bool = False,       # rebuild cache if True
                 use_node_attr: bool = False,      # accepted for compat; ignored
                 N: int = 10_000,
                 seed: int = 0,
                 **kwargs):
        self.N = int(N)
        self.seed = int(seed)

        # Unique cache per (N, seed)
        self._processed_fname = f"synthetic_triad_N{self.N}_seed{self.seed}.pt"
        processed_path = os.path.join(root, "processed", self._processed_fname)

        if force_reload and os.path.exists(processed_path):
            os.remove(processed_path)

        # Let InMemoryDataset handle process() call if missing:
        super().__init__(root, transform, pre_transform)

        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            # older torch without weights_only kw
            self.data, self.slices = torch.load(self.processed_paths[0])


    def __getitem__(self, idx: Union[int, slice, List[int]]):
        if isinstance(idx, int):
            return InMemoryDataset.get(self, idx)
        return super().__getitem__(idx)           


    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return [self._processed_fname]

    @property
    def num_classes(self): return 4

    @property
    def num_features(self): return 2

    def download(self): pass

    def process(self):
        _set_seed(self.seed)
        graphs: List[Data] = []
        tri_edges = [(0, 1), (0, 2), (1, 2)]

        for _ in range(self.N):
            pts = torch.tensor([_sample_unit_disk_point() for _ in range(3)], dtype=torch.float)

            # y=0: no edges (empty edge_index)
            e0 = torch.empty((2, 0), dtype=torch.long)
            graphs.append(Data(x=pts, edge_index=e0, y=torch.tensor([0], dtype=torch.long), num_nodes=3))

            # choose a random ordering of the 3 possible edges once per triple
            perm = tri_edges[:]
            random.shuffle(perm)

            # y=1: one edge
            e1 = torch.tensor(perm[0], dtype=torch.long).view(2, 1)
            e1 = to_undirected(e1, num_nodes=3)
            graphs.append(Data(x=pts, edge_index=e1, y=torch.tensor([1], dtype=torch.long), num_nodes=3))

            # y=2: two edges
            e2 = torch.tensor(perm[:2], dtype=torch.long).t().contiguous()
            e2 = to_undirected(e2, num_nodes=3)
            graphs.append(Data(x=pts, edge_index=e2, y=torch.tensor([2], dtype=torch.long), num_nodes=3))

            # y=3: full triangle
            e3 = torch.tensor(tri_edges, dtype=torch.long).t().contiguous()
            e3 = to_undirected(e3, num_nodes=3)
            graphs.append(Data(x=pts, edge_index=e3, y=torch.tensor([3], dtype=torch.long), num_nodes=3))

        # optional pre_transform (your SYN branch sets this to Compose([]), so it's a no-op)
        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
