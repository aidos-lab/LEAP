import dect
import dect.directions
import torch
import torch.nn.functional as F
from models.baseGraphModel import BaseGraphModel
from utils.utility import ConvType
from dect.nn import ECTLayer

class GCNwithLECTallSteps(BaseGraphModel):
	
	def __init__(
		self,
		in_channels,
		out_channels,
		hidden_channels,
		num_layers,
		ect_directions,
		conv_name="GCNConv",
		**kwargs,
	):
		# Set use_ect to True as it is always used in this model
		kwargs['use_ect'] = True
		kwargs['ect_directions'] = ect_directions
		super(GCNwithLECTallSteps, self).__init__(in_channels, out_channels, **kwargs)

		# get the convolution class corresponding to the received conv_name
		Conv = ConvType.convert(conv_name)

		# Warning!!
		# ECT embed dim was 5 by default here, before the use of BaseGraphModel

		self.convs = torch.nn.ModuleList() # Graph Convolutions
		self.ect_layers = torch.nn.ModuleList() # l-ECT is computed at each GNN step
		self.ect_embed_layers = torch.nn.ModuleList() # l-ECT is projected at each GNN step
		

		# Graph Conv Layers
		self.convs.append(Conv(self.num_feat_1st_layer, hidden_channels - self.ect_embed_dim))
		for _ in range(num_layers - 2):
			self.convs.append(Conv(hidden_channels, hidden_channels - self.ect_embed_dim))
		self.convs.append(Conv(hidden_channels, out_channels))

		# Add l-ECT layers, the 1st one was already created by the parent class
		for _ in range(num_layers -1):
			# l-ECT layer
			self.ect_layers.append(
				ECTLayer(self.ect_config,
					dect.directions.generate_uniform_directions(
						num_thetas=self.num_thetas,
						d = hidden_channels - self.ect_embed_dim,
						seed=self.ect_seed,
						device=self.ect_directions.device)
				)
			)
			# l-ECT projection layers
			self.ect_embed_layers.append(self._build_ect_embed_layer(pe_dim = hidden_channels - self.ect_embed_dim))

	def forward(self, x, e_index, **kwargs):
		
		# Gather initial node features concatenated with l-ECT (and PE if applicable)
		x = self._common_initial_forward(x, e_index, **kwargs)
		
		# Go through graph conv layers
		for i in range(len(self.convs) - 1):
			# Graph conv
			conv_out = F.relu(self.convs[i](x, e_index)) 
			
			# Compute l-ECT on conv output and project
			ect_features = self._compute_lect_batch_wise(x=conv_out, e_index=e_index, ect_layer=self.ect_layers[i], **kwargs)
			ect_features = self.ect_embed_layers[i](ect_features, directions = self.ect_layers[i].v)

			# Concatenate conv output and l-ECT
			x = torch.cat([conv_out, ect_features], axis=1)
		
		# Apply final conv layer
		x = self.convs[-1](x, e_index)
		return x
