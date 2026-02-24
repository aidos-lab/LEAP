from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import fields
from typing import Optional

@dataclass
class Config:
    dataset_name: str = 'Letter-high'
    batch_size: int = 16
    max_epochs: int = 100
    exclude_features: bool = False
    learnable_pe: bool = False
    wandb_name: str = ""
    fold: int = 3
    seed: int = 42
    model_name: str = 'GCN' # GCN | NoMP | GCNwithLECTallSteps

    hidden_dim: Optional[int] = 32 # Only for GNN not for transformer arch
    num_layers: Optional[int] = 5 # Only for GNN not for transformer arch
    conv_name: Optional[str] = "GCNConv" # GCNConv | GATConv, Only for GNN not for transformer arch

    use_ect: bool = True
    ect_type: Optional[str] = 'edges' # edges | points,  Only makes sense if use_ect is True
    ect_hops: Optional[tuple[int]] = (1,) # Only makes sense if use_ect is True
    num_thetas: Optional[int] = 16 # Only makes sense if use_ect is True
    ect_seed: Optional[int] = 0 # Only makes sense if use_ect is True
    ect_on_pe: Optional[bool] = False # Only makes sense if use_ect is True
    learn_directions: Optional[bool] = False # Only makes sense if use_ect is True
    ect_embed_method: Optional[str] = "linear" # Only makes sense if use_ect is True
    ect_embed_dim: Optional[int] = 10 # Only makes sense if use_ect is True
    ect_scale: Optional[int] = 128
    ect_resolution: Optional[int] = 16
    ect_radius: Optional[float] = 1.1
    ect_normalize_before: Optional[bool] = True
    ect_normalize_by_graph: Optional[bool] = False

    use_pos_enc: bool = False 
    pos_enc_dim: Optional[int] = 10 # Only makes sense if use_pos_enc is True
    pos_enc_embed_dim: Optional[int] = 10 # Only makes sense if use_pos_enc is True
    pe_name: Optional[str] = 'rwpe' # rwpe | lape, Only makes sense if use_pos_enc is True

    def to_dict(self) -> dict:
        return asdict(self)
    
    def filter_dict(unfiltered_dict) -> dict:
        valid_keys = {f.name for f in fields(Config)}
        filtered_dict = {k: v for k, v in unfiltered_dict.items() if k in valid_keys}
        return filtered_dict


