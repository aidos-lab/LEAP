import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
from models.baseGraphModel import BaseGraphModel
from utils.utility import ConvType

class MyGNN(BaseGraphModel):
	def __init__(
		self,
		in_channels,
		out_channels,
		hidden_channels,
		num_layers,
		conv_name="GCNConv",
		**kwargs,
	):
		super(MyGNN, self).__init__(in_channels, out_channels, **kwargs)

		Conv = ConvType.convert(conv_name)
		
		self.convs = torch.nn.ModuleList()
		
		self.learnable_pe = kwargs.get("learnable_pe", False)

		if self.learnable_pe:
			self.learned_pe_dim = kwargs.get("learned_pe_dim", 5)
			self.pe_convs = self.pe_convs = torch.nn.ModuleList()
			
			# Learnable PE Layers
			self.pe_convs.append(Conv(self.num_feat_1st_layer - in_channels, self.learned_pe_dim))
			for _ in range(num_layers - 2):
				self.pe_convs.append(Conv(self.learned_pe_dim, self.learned_pe_dim))

		else:
			self.learned_pe_dim = 0
		

		# Graph Conv Layers
		self.convs.append(Conv(self.num_feat_1st_layer, hidden_channels - self.learned_pe_dim))
		for _ in range(num_layers - 2):
			self.convs.append(Conv(hidden_channels, hidden_channels - self.learned_pe_dim))
		self.convs.append(Conv(hidden_channels, out_channels))


	def forward(self, x, e_index, **kwargs):

		# Handle computation of ECT/PE, projections, concatenation etc.
		x = self._common_initial_forward(x, e_index, **kwargs)
	
		# pass through the GCN layers
		if self.learnable_pe:
			return self._apply_convs_with_learnable_pe(x, e_index)

		return self._apply_convs(x, e_index)

	def _apply_convs(self, x, e_index):
		for conv in self.convs[:-1]:
			x = F.relu(conv(x, e_index))
		
		x = self.convs[-1](x, e_index)
		return x
	
	def _apply_convs_with_learnable_pe(self, x, e_index):

		# Separate received features into PE and x
		pe = x[:, self.in_channels:]
		x = x[:, :self.in_channels]

		for i in range(len(self.pe_convs) - 1):
			# update x based on x and PEs
			x = F.relu(self.convs[i](torch.cat([x, pe], axis = 1), e_index))
			# update learnable PEs based only on PEs
			pe = F.tanh(self.pe_convs[i](pe, e_index))
		
		# final x based on last x concatenated with final PEs
		x = self.convs[-1](torch.cat([x, pe], axis = 1), e_index)

		return x