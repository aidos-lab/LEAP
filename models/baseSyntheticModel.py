from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from dect.nn import ECTConfig, ECTLayer

from models.ECTProjections import (
    Conv1dEquivariant,
    DeepSets,
    LinearProj,
    SetTransformer,
)
from utils.utility import ECTembedMethod, ECTtype, build_neigh_batch, maxL2_graph_norm


class BaseSyntheticModel(nn.Module, ABC):
    """Abstract class for Graph Models.
    - Handles common argument parsing when instantiating graph models.
    - Implements common methods for handling the l-ECT computation
    """

    in_channels: int
    out_channels: int
    num_feat_1st_layer: int
    verbose: bool
    exclude_features: bool
    use_pos_enc: bool
    pos_enc_embed_dim: Optional[int]
    use_ect: bool
    ect_on_pe: Optional[bool]
    ect_embed_dim: Optional[bool]
    ect_directions: Optional[torch.Tensor]
    ect_resolution: Optional[int]
    ect_radius: Optional[float]
    ect_scale: Optional[float]
    ect_type: Optional[ECTtype]
    ect_normalize_before: Optional[bool]
    ect_normalize_after: Optional[bool]
    ect_hops: Optional[int]
    learn_directions: Optional[bool]
    ect_embed_method: Optional[ECTembedMethod]
    normalize_by_graph: Optional[bool]

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # Set in and out channels, all models have this parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Originally set the number of features for the first layer to in_channels
        # This may change depending on the configuration (i.e. increase if l-ECT or pe is concatenated)
        self.num_feat_1st_layer = self.in_channels

        # Set all the attributes passed as kwargs
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Set default values for non passed args
        for attr, value in BaseSyntheticModel.default_values().items():
            if attr not in kwargs:
                setattr(self, attr, value)

        self._print("Instantiated Model in verbose mode")

        if self.exclude_features:
            self.num_feat_1st_layer = 0
            self._print("Original node features wont be passed to the model")

        if self.use_ect:
            self._print("ECT will be appended as node features")

            if self.ect_on_pe:
                self._print("ECT will be computed using PE instead of node features")

            self.num_thetas = self.ect_directions.shape[1]
            self._print(f"ECT will be computed with {self.num_thetas} directions")

            self._print(f"ECT will be computed with {self.ect_resolution} resolution")
            self._print(f"ECT will be computed with {self.ect_scale} scale")
            self._print(f"ECT will be computed in {self.ect_type} mode")
            self._print(f"ECT will be computed with {self.ect_radius} radius")

            if self.ect_normalize_before:
                self._print(
                    "Features fed to ECT will be normalized by mean substraction and division by max L2 norm"
                )

            if self.ect_normalize_after:
                self._print("Results of ECT will be normalized")

            self.num_feat_1st_layer += self.ect_embed_dim * len(self.ect_hops)
            self.ect_embed_layer = self._build_ect_embed_layer()
            self._print(f"ECT results will be projected to dim {self.ect_embed_dim}")
            self._print(f"{self.ect_embed_method} will be used for projection of ECT")

            self._print(
                f'ECT directions are {"learnable" if self.learn_directions else "fixed"}'
            )

            self.ect_config = ECTConfig(
                ect_type=self.ect_type,
                resolution=self.ect_resolution,
                scale=self.ect_scale,
                radius=self.ect_radius,
                normalized=self.ect_normalize_after,
                fixed=not self.learn_directions,
            )
            self.ect_layer = ECTLayer(self.ect_config, v=self.ect_directions)

        if self.use_pos_enc:
            self.num_feat_1st_layer += self.pos_enc_embed_dim

            self._print("Using PE aside of l-ECT")
            self._print(f"Dimension of PE is {self.pos_enc_dim}")

            if self.pos_enc_embed_dim != self.pos_enc_dim:
                self._print(f"PE will be projected to dim {self.pos_enc_embed_dim}")

                self.pos_enc_embed_layer = nn.Linear(
                    self.pos_enc_dim, self.pos_enc_embed_dim
                )
            else:
                self.pos_enc_embed_layer = None

        self.mlp = torch.nn.Sequential(
            nn.Linear(self.ect_embed_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, batch):

        # ect
        x = self.ect_layer(batch)
        x = self.ect_embed_layer(x, directions=self.ect_layer.v)
        x = self.mlp(x)

        return x

    def _build_ect_embed_layer(self, **kwargs):
        if self.ect_embed_method == ECTembedMethod.LINEAR.value:
            return LinearProj(
                input_size=self.ect_resolution * self.num_thetas,
                output_size=self.ect_embed_dim,
            )

        if self.ect_embed_method == ECTembedMethod.ATTN.value:
            return SetTransformer(
                feature_dim=self.ect_resolution, out_dim=self.ect_embed_dim
            )

        if self.ect_embed_method == ECTembedMethod.ATTN_PE.value:
            pe_dim = kwargs.get("pe_dim", self.ect_directions.shape[0])
            return SetTransformer(
                feature_dim=self.ect_resolution,
                out_dim=self.ect_embed_dim,
                pe_dim=pe_dim,
            )

        if self.ect_embed_method == ECTembedMethod.DEEPSETS.value:
            return DeepSets(
                in_channels=self.ect_resolution,
                out_channels=self.ect_embed_dim,
            )

        if self.ect_embed_method == ECTembedMethod.CONV1D.value:
            return Conv1dEquivariant(
                num_thetas=self.num_thetas,
                num_steps=self.ect_resolution,
                output_size=self.ect_embed_dim,
            )

        raise NotImplementedError(
            f"Method {self.ect_embed_method} not implemented for ect projection"
        )

    def _print(self, message: str):
        if self.verbose:
            print(message)

    @staticmethod
    def default_values() -> dict:
        return {
            "verbose": False,
            "exclude_features": False,
            "use_pos_enc": False,
            "pos_enc_embed_dim": 10,
            "use_ect": False,
            "ect_on_pe": False,
            "ect_embed_dim": 10,
            "ect_directions": None,
            "ect_resolution": 16,
            "ect_radius": 1.1,
            "ect_scale": 128,
            "ect_type": "edges",
            "ect_normalize_before": True,
            "ect_normalize_after": False,
            "ect_hops": 1,
            "learn_directions": False,
            "ect_embed_method": "linear",
        }

    @staticmethod
    def get_kwarg_default(kwarg: str) -> bool | int | str | None:
        if kwarg in BaseSyntheticModel.default_values:
            return BaseSyntheticModel.default_values[kwarg]

        raise KeyError(f"Unknown kwarg: '{kwarg}'")

