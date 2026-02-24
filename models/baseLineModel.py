import torch
from torch_geometric.nn.pool import global_add_pool
from models.gcn import MyGNN
from models.gCNwithLECTallSteps import GCNwithLECTallSteps
from models.noMP import NoMP
from models.baseSyntheticModel import BaseSyntheticModel

class BaseLineModel(torch.nn.Module):
    def __init__(
        self, input_dim, num_classes, hidden_dim, num_layers, model_name="GCN", regression = False, **kwargs
    ):
        super().__init__()
        self.regression = regression
        # HIV dataset needs to handle that features are categorical.
        # We do this outside the model, so that it's implemented here for all models.
        # Also this ensures that the l-ect are computed in a "continuous" space.
        self.embedding_layers = None
        if kwargs.get('num_embed_feat'):
            self.embedding_layers = torch.nn.ModuleList()
            self.embedding_layers_idx = []
            computed_input_dim = 0
            for i in range(len(kwargs['num_embed_feat'])):
                num_embeddings = kwargs['num_embed_feat'][i]
                embedding_dim = num_embeddings // 2
                computed_input_dim += embedding_dim
                if embedding_dim != 0:
                    self.embedding_layers_idx.append(len(self.embedding_layers))
                    self.embedding_layers.append(torch.nn.Embedding(num_embeddings, embedding_dim))
                else:
                    self.embedding_layers_idx.append(None)
            print(f'computed_input_dim: {computed_input_dim}')

            if kwargs['use_ect']:
                self.mlp_proj = torch.nn.Sequential(
                    torch.nn.Linear(computed_input_dim, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 3)
		        )

                # self.aux_decoder = torch.nn.Sequential(
                #     torch.nn.Linear(3, 16),
                #     torch.nn.ReLU(),
                #     torch.nn.Linear(16, computed_input_dim)  # same as original high-dim vector
                # )

                    
                
            #assert computed_input_dim == input_dim, f"Input dim {input_dim} does not match computed input dim {input_dim}"

        # Select the model based on the model_name
        if model_name == "GCN":
            self.model = MyGNN(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers,
                **kwargs,
            )
        elif model_name == "GCNwithLECTallSteps":
            self.model = GCNwithLECTallSteps(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers,
                **kwargs,
            )
        elif model_name == "NoMP":
            self.model = NoMP(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers,
                **kwargs,
            )
        elif model_name == "SYN":
            self.model = BaseSyntheticModel(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.name = model_name



    def forward(self, data):
        if self.name == "SYN":
            return torch.nn.functional.log_softmax(self.model(data), dim=-1)
        
        x, e_index = data.x, data.edge_index
        x_ect = None
        if self.embedding_layers is not None:
            embeddings = []
            for i in range(x.shape[1]):
                idx = self.embedding_layers_idx[i]
                if idx is not None:
                    embeddings.append(self.embedding_layers[idx](x[:, i]))
            x = torch.cat(embeddings, dim=1)
            aux_x = x.clone()
            if self.model.use_ect:
                x_ect = self.mlp_proj(aux_x)
                #aux_target = x_ect.detach() 
                pass 

        pos_enc = data.get('pos_enc', None)
        
        x = self.model(x, e_index, pos_enc=pos_enc, batch=data.batch, subgraph_info=data.get('subgraph_info',None), ptr=data.ptr, coords4ect = x_ect)
        x = global_add_pool(x, data.batch, size=len(data))
        
        if self.regression:
            return x

        if False and self.model.use_ect and self.embedding_layers is not None:
            reconstructed = self.aux_decoder(aux_target)
            return torch.nn.functional.log_softmax(x, dim=-1), aux_x, reconstructed
        return torch.nn.functional.log_softmax(x, dim=-1)
