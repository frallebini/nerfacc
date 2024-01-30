import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import math
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from itertools import permutations, product
from torch import Tensor
from tqdm import tqdm
from typing import Any, Dict

from datasets.shapenet import ShapeNetLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from radiance_fields.ngp_single_mlp import NGPRadianceFieldSingleMlp
from utils import render_image_with_occgrid_test


def get_state_dict(cfg: Dict[str, Any]) -> Dict[str, Tensor]:
    sd_A = torch.load(f"{cfg['cfg_dir']}/{cfg['id']}_A.pt")
    sd_B = torch.load(f"{cfg['cfg_dir']}/{cfg['id']}_B.pt")
    if cfg["mlp"]["use_single"]:
        sd_A["radiance_field"]["mlp.params"] = sd_B["radiance_field"]["mlp.params"]
    else:
        sd_A["radiance_field"]["mlp_base.1.params"] = sd_B["radiance_field"]["mlp_base.1.params"]  # density mlp
        sd_A["radiance_field"]["mlp_head.params"] = sd_B["radiance_field"]["mlp_head.params"]  # color mlp

    return sd_A


def search_perm(cfg: Dict[str, Any]) -> None:
    id = cfg["id"]
    cfg_dir = cfg["cfg_dir"]
    
    encoding = cfg["encoding"]["type"]
    n_levels = cfg["encoding"]["n_levels"]
    n_features_per_level = cfg["encoding"]["n_features_per_level"]
    base_resolution = cfg["encoding"]["base_resolution"]
    max_resolution = cfg["encoding"]["max_resolution"]

    activation = cfg["mlp"]["activation"]
    use_single_mlp = cfg["mlp"]["use_single"]
    use_viewdirs = cfg["mlp"]["use_viewdirs"]
    if use_single_mlp:
        n_neurons = cfg["mlp"]["n_neurons"]
        n_hidden_layers = cfg["mlp"]["n_hidden_layers"]

    # dataset parameters
    data_root = cfg["data_root"]
    # scene parameters
    aabb = torch.tensor([-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]).cuda()
    near_plane = 0.0
    far_plane = 1.0e10
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_n_samples = 1024
    render_step_size = (
        (aabb[3:] - aabb[:3]).max() * math.sqrt(3) / render_n_samples
    ).item()
    alpha_thre = 0.0
    cone_angle = 0.0

    test_dataset = ShapeNetLoader(
        subject_id=id,
        root_fp=data_root,
        split="train",
        num_rays=None,
        device="cuda:0"
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, 
        resolution=grid_resolution, 
        levels=grid_nlvl
    ).cuda()
    
    if use_single_mlp:
        radiance_field = NGPRadianceFieldSingleMlp(
            aabb=estimator.aabbs[-1],
            use_viewdirs=use_viewdirs,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            encoding_type=encoding,
            mlp_activation=activation,
            n_neurons=n_neurons,
            n_hidden_layers=n_hidden_layers
        ).cuda()
    else:
        radiance_field = NGPRadianceField(
            aabb=estimator.aabbs[-1],
            use_viewdirs=use_viewdirs,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            encoding_type=encoding,
            mlp_activation=activation,
        ).cuda()

    psnr_best = 0
    perm_tuples = list(product(permutations(range(n_features_per_level)), repeat=n_levels))

    for step, perms in enumerate(tqdm(perm_tuples)):
        # reset state dict
        sd = get_state_dict(cfg)

        for l in range(n_levels):
            # get grid
            if use_single_mlp:
                grid = sd["radiance_field"][f"encoding.levels.{l}.embedding.weight"]
            else:
                grid = sd["radiance_field"][f"mlp_base.0.levels.{l}.embedding.weight"]

            # permute grid
            grid = torch.index_select(grid, dim=1, index=torch.IntTensor(perms[l]).cuda())

            # set grid
            if use_single_mlp:
                sd["radiance_field"][f"encoding.levels.{l}.embedding.weight"] = grid
            else:
                sd["radiance_field"][f"mlp_base.0.levels.{l}.embedding.weight"] = grid
                
        estimator.load_state_dict(sd["estimator"])
        radiance_field.load_state_dict(sd["radiance_field"])

        # evaluation
        estimator.eval()
        radiance_field.eval()

        psnrs = []
        with torch.no_grad():
            for i in range(len(test_dataset)):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                try:
                    # rendering
                    rgb, _, _, _ = render_image_with_occgrid_test(
                        render_n_samples,
                        # scene
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                    )
                except RuntimeError:
                    tqdm.write(f"skipped test sample {i} permutation {perms}")
                    continue

                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
            
        psnr = sum(psnrs) / len(psnrs)
        wandb.log({"test/psnr": psnr}, step=step)
        
        if psnr > psnr_best:
            psnr_best = psnr
            wandb.log({"test/psnr_best": psnr_best}, step=step)
            torch.save(sd, f"{cfg_dir}/{id}_perm.pt")


if __name__ == "__main__":
    ##################################################
    id = "02958343/1c53bc6a3992b0843677ee89898ae463"
    cfg_dir = "ckpts/torch_relu/shapenet"
    ###################################################

    with open(f"{cfg_dir}/cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["id"] = id
    cfg["cfg_dir"] = cfg_dir

    encoding = cfg["encoding"]["type"]
    n_levels = cfg["encoding"]["n_levels"]
    n_features_per_level = cfg["encoding"]["n_features_per_level"]
    base_resolution = cfg["encoding"]["base_resolution"]

    activation = cfg["mlp"]["activation"]
    use_single_mlp = cfg["mlp"]["use_single"]
    use_viewdirs = cfg["mlp"]["use_viewdirs"]
    if use_single_mlp:
        n_neurons = cfg["mlp"]["n_neurons"]
        n_hidden_layers = cfg["mlp"]["n_hidden_layers"]

    run_name = \
        f"{id}_search_perm" + \
        f"_{encoding}_{activation.lower()}" + \
        f"{'_single' if use_single_mlp else ''}" + \
        f"{'_viewdir' if use_viewdirs else ''}" + \
        f"_{n_levels}_{n_features_per_level}_{base_resolution}"

    wandb.init(
        entity="frallebini",
        project="nerfacc",
        name=run_name,
        config=cfg
    )
    
    search_perm(cfg)
