from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
import torch
from typing import Union
from torch_geometric.transforms import BaseTransform

class PrecomputeOneHopStructureOnly(BaseTransform):
	def __init__(self, num_hops: Union[int, list[int]] = [1]):
		super().__init__()
		if isinstance(num_hops, int):
			num_hops = [num_hops]
		self.num_hops = num_hops
		print(self.num_hops)

	def __call__(self, data):
		subgraph_info = dict()
		
		for k in self.num_hops:
			k_hop_info = []
			for i in range(data.num_nodes):
				subset, edge_index, _, _ = k_hop_subgraph(
					node_idx=i,
					num_hops=k,
					edge_index=data.edge_index,
					relabel_nodes=True,
					num_nodes=data.num_nodes
				)
				k_hop_info.append({
				'subset': subset,
				'edge_index': edge_index
				})
			subgraph_info[k] = k_hop_info

		data.subgraph_info = subgraph_info
		return data

	def _old__call__(self, data):
		subgraph_info = []

		for i in range(data.num_nodes):
			subset, edge_index, _, _ = k_hop_subgraph(
				node_idx=i,
				num_hops=self.num_hops,
				edge_index=data.edge_index,
				relabel_nodes=True,
				num_nodes=data.num_nodes
			)

			subgraph_info.append({
				'subset': subset,
				'edge_index': edge_index
			})

		data.subgraph_info = subgraph_info
		return data

def _custom_collate_fn(batch):
	# Save subgraph_info separately
	subgraph_infos = [data.subgraph_info for data in batch]

	# Remove subgraph_info before batching to avoid conflicts
	for data in batch:
		del data.subgraph_info

	# Perform standard PyG batching
	batched_data = Batch.from_data_list(batch)

	# Attach subgraph_info back as a regular Python attribute
	setattr(batched_data, 'subgraph_info', subgraph_infos)

	return batched_data

def custom_collate_fn(batch):
	print('[SAKLJFAKLJF;ASDLKJF;KJF;ASJKF]')
	print('[SAKLJFAKLJF;ASDLKJF;KJF;ASJKF]')
	print('[SAKLJFAKLJF;ASDLKJF;KJF;ASJKF]')
	all_subgraph_infos = [data.subgraph_info for data in batch]

	for data in batch:
		del data.subgraph_info

	batched_data = Batch.from_data_list(batch)
	x = batched_data.x
	ptr = batched_data.ptr

	subgraph_batches = {}

	if isinstance(all_subgraph_infos[0], list):
		print('going through first if in custom collate')
		all_subgraphs = []
		for graph_idx, subgraph_info in enumerate(all_subgraph_infos):
				k_hop_info = subgraph_info
				node_offset = ptr[graph_idx]

				for info in k_hop_info:
					subset = info['subset']
					edge_index = info['edge_index']
					global_subset = subset + node_offset
					neighbors = x[global_subset]

					all_subgraphs.append(Data(x=neighbors, edge_index=edge_index))

		subgraph_batches[1] = Batch.from_data_list(all_subgraphs)
		batched_data.subgraph_batches = subgraph_batches
	
	else:
		hop_keys = all_subgraph_infos[0].keys()  # assume all share the same hop list
		for k in hop_keys:
			all_subgraphs = []

			for graph_idx, subgraph_info_dict in enumerate(all_subgraph_infos):
				k_hop_info = subgraph_info_dict[k]
				node_offset = ptr[graph_idx]

				for info in k_hop_info:
					subset = info['subset']
					edge_index = info['edge_index']
					global_subset = subset + node_offset
					neighbors = x[global_subset]

					all_subgraphs.append(Data(x=neighbors, edge_index=edge_index))

			subgraph_batches[k] = Batch.from_data_list(all_subgraphs)
		batched_data.subgraph_batches = subgraph_batches
	return batched_data



device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

if __name__ == "__main__":
	print("instantiating dataset")
	dataset = TUDataset(name="Letter-high", root="./data/TU", transform=PrecomputeOneHopStructureOnly(1))

	print("instantiating loader")
	loader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, pin_memory=True)

	for batch in loader:
		batch = batch.to(device)
		x = batch.x
		ptr = batch.ptr  # tensor of shape [batch_size + 1]

		all_subgraphs = []

		# Flatten all subgraph info across all graphs in batch
		for graph_idx, subgraph_info_list in enumerate(batch.subgraph_info):
			node_offset = ptr[graph_idx]

			# Pre-batch subsets and edge_index tensors
			for info in subgraph_info_list:
				subset = info['subset'].to(device)
				edge_index = info['edge_index'].to(device)

				global_subset = subset + node_offset

				neighbors = x[global_subset]
				# neighbors = neighbors - neighbors.mean(dim=0, keepdim=True)
				# neighbors = neighbors / (neighbors.norm(dim=1).max() + 1e-6)

				all_subgraphs.append(Data(x=neighbors, edge_index=edge_index))

		subgraph_batch = Batch.from_data_list(all_subgraphs).to(device)

		batch_list = []
		for i in range(x.shape[0]):
			subset, edge_index, _, _ = k_hop_subgraph(
				node_idx=i,
				num_hops=1,
				edge_index=batch.edge_index,
				relabel_nodes=True,
				num_nodes=x.shape[0],
			)
			neighbors = x[subset]
			# neighbors = neighbors - neighbors.mean(dim=0, keepdim=True)
			# neighbors = neighbors / (neighbors.norm(dim=1).max() + 1e-6)

			data = Data(
				x=neighbors,
				edge_index=edge_index,
			)
			batch_list.append(data)
		batch_to_compare = Batch.from_data_list(batch_list)

		print("------")
		print(f"Main batch: {batch}")
		print(f"Subgraph batch: {subgraph_batch}")
		print(f"Second subgraph_batch: {batch_to_compare}")
		batch1 = subgraph_batch
		batch2 = batch_to_compare
		batch1.to(device)
		batch2.to(device)
		torch.testing.assert_close(batch1.x, batch2.x)
		torch.testing.assert_close(batch1.edge_index, batch2.edge_index)
		torch.testing.assert_close(batch1.batch, batch2.batch)
		print("------")