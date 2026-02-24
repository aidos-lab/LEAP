from torch_geometric.utils import  to_dense_batch
import torch

from models.baseGraphModel import BaseGraphModel

class NoMP(BaseGraphModel):
	def __init__(
		self,
		in_channels,
		out_channels,
		**kwargs,
	):
		super(NoMP, self).__init__(in_channels, out_channels, **kwargs)

		self.nhead = kwargs.get("nhead", 1)
		self.num_trans_layers = kwargs.get("num_trans_layers", 1)
		self.dim_feedforward = kwargs.get("dim_feedforward", 64)
		self.d_model = kwargs.get("d_model", 16)
		
		self.input_proj = torch.nn.Linear(self.num_feat_1st_layer, self.d_model) if self.num_feat_1st_layer != self.d_model else torch.nn.Identity()
		
		encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True, dim_feedforward=self.dim_feedforward)
		self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_trans_layers)
		self.to_class_layer = torch.nn.Linear(self.d_model, out_channels)

	def forward(self, x, e_index, **kwargs):
		# Handle computation of ECT/PE, projections, concatenation etc.
		x = self._common_initial_forward(x, e_index, **kwargs)

		# Project or apply identity
		x = self.input_proj(x)
		
		# Create dense batch for efficient forward in transformer
		batch = kwargs["batch"]
		x_dense, mask = to_dense_batch(x, batch)
		transformer_mask = ~mask

		# Forward through transformer
		x_encoded = self.transformer_encoder(x_dense, src_key_padding_mask=transformer_mask) 

		# Forward to get proper output dimension
		x_encoded = self.to_class_layer(x_encoded)

		# Discard padding
		x_out = x_encoded[mask]

		return x_out
