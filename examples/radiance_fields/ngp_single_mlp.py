"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import numpy as np
import random
import sys
import torch

from typing import Callable, List, Union

from radiance_fields.encoding import MultiResHashGrid
from radiance_fields.ngp import _TruncExp, contract_to_unisphere

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


trunc_exp = _TruncExp.apply


class NGPRadianceFieldSingleMlp(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        encoding_type: str = "combo",
        mlp_activation: str = "ReLU",
        n_neurons: int = 64,
        n_hidden_layers: int = 3
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.encoding_type = encoding_type
        self.mlp_activation = mlp_activation
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "n_dims_to_encode": 3,
                    "degree": 4,
                },
                seed=random.randint(0, sys.maxsize),
            )

        encoding_config={
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        if encoding_type == "cuda":
            # separate Tiny CUDA encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=num_dim, 
                encoding_config=encoding_config,
                seed=random.randint(0, sys.maxsize),
            )
        elif encoding_type == "torch":
            # separate pure PyTorch encoding
            self.encoding = MultiResHashGrid(
                num_dim,
                encoding_config["n_levels"],
                encoding_config["n_features_per_level"],
                encoding_config["log2_hashmap_size"],
                encoding_config["base_resolution"],
                max_resolution,
            )
        else:
            raise Exception("Unsupported encoding type")
        
        mlp_type = "CutlassMLP" if mlp_activation == "Sine" else "FullyFusedMLP"
        self.mlp = tcnn.Network(
            n_input_dims=
                (self.encoding.n_output_dims if encoding_type == "cuda" else self.encoding.output_dim) +
                (self.direction_encoding.n_output_dims if use_viewdirs else 0),
            n_output_dims=4,
            network_config={
                "otype": mlp_type,
                "activation": mlp_activation,
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
            seed=random.randint(0, sys.maxsize),
        )
    
    def _query_rgb_and_density(self, x, dir=None):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x_emb = self.encoding(x)

        if self.use_viewdirs:
            if dir is not None:
                dir = (dir + 1.0) / 2.0
                d_emb = self.direction_encoding(dir)
            else:
                d_emb = torch.rand(x.shape[0], self.direction_encoding.n_output_dims).to(x_emb)
            mlp_in = torch.cat((x_emb, d_emb), dim=-1)
        else:
            mlp_in = x_emb
            
        mlp_out = self.mlp(mlp_in)

        rgb, density_before_activation = mlp_out[..., :3], mlp_out[..., 3]
        density_before_activation = density_before_activation[:, None]

        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        rgb = torch.sigmoid(rgb)

        return rgb, density
    
    def query_density(self, x):
        _, density = self._query_rgb_and_density(x)
        
        return density

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        rgb, density = self._query_rgb_and_density(positions, directions)

        return rgb, density
