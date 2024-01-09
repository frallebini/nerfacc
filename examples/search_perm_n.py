import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from torch import Tensor
from tqdm import tqdm
from typing import Any, Dict

from datasets.nerf_synthetic import SubjectLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp_single_mlp import NGPRadianceFieldSingleMlp
from utils import render_image_with_occgrid_test


def get_state_dict(cfg: Dict[str, Any]) -> Dict[str, Tensor]:
    sd_A = torch.load(f"{cfg['cfg_dir']}/{cfg['scene']}_A.pt")
    sd_B = torch.load(f"{cfg['cfg_dir']}/{cfg['scene']}_B.pt")
    sd_A["radiance_field"]["mlp.params"] = sd_B["radiance_field"]["mlp.params"]

    return sd_A


def search_perm(cfg: Dict[str, Any]) -> None:
    scene = cfg["scene"]
    sample_idx = cfg["sample_idx"]
    cfg_dir = cfg["cfg_dir"]
    
    encoding = cfg["encoding"]["type"]
    n_levels = cfg["encoding"]["n_levels"]
    n_features_per_level = cfg["encoding"]["n_features_per_level"]
    base_resolution = cfg["encoding"]["base_resolution"]

    activation = cfg["mlp"]["activation"]
    use_viewdirs = cfg["mlp"]["use_viewdirs"]
    n_neurons = cfg["mlp"]["n_neurons"]
    n_hidden_layers = cfg["mlp"]["n_hidden_layers"]

    # dataset parameters
    data_root = cfg["data_root"]
    test_dataset_kwargs = {}
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]).cuda()
    near_plane = 0.0
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=data_root,
        split="test",
        num_rays=None,
        device="cuda:0",
        **test_dataset_kwargs,
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, 
        resolution=grid_resolution, 
        levels=grid_nlvl
    ).cuda()
    
    radiance_field = NGPRadianceFieldSingleMlp(
        aabb=estimator.aabbs[-1],
        use_viewdirs=use_viewdirs,
        base_resolution=base_resolution,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        encoding_type=encoding,
        mlp_activation=activation,
        n_neurons=n_neurons,
        n_hidden_layers=n_hidden_layers
    ).cuda()

    psnr_best = 0
    perms = itertools.permutations(range(n_features_per_level))
    n_perms = math.factorial(n_features_per_level)

    for step, perm in enumerate(tqdm(perms, total=n_perms)):
        # reset state dict
        sd = get_state_dict(cfg)

        # get grid
        grid = sd["radiance_field"]["encoding.levels.0.embedding.weight"]

        # permute grid
        grid = torch.index_select(grid, dim=1, index=torch.IntTensor(perm).cuda())

        # set grid
        sd["radiance_field"][f"encoding.levels.0.embedding.weight"] = grid
        estimator.load_state_dict(sd["estimator"])
        radiance_field.load_state_dict(sd["radiance_field"])

        # evaluation
        estimator.eval()
        radiance_field.eval()

        with torch.no_grad():
            data = test_dataset[sample_idx]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            try:
                # rendering
                rgb, _, _, _ = render_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
            except RuntimeError:
                tqdm.write(f"skipped permutation {perm}")
                continue

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            wandb.log({"test/psnr": psnr}, step=step)
        
        if psnr > psnr_best:
            psnr_best = psnr
            wandb.log({"test/psnr_best": psnr_best}, step=step)
            torch.save(sd, f"{cfg_dir}/{scene}_perm.pt")


if __name__ == "__main__":
    ##########################################
    scene = "chair"
    sample_idx = 42
    cfg_dir = "ckpts/torch_sine_single_1_16_32"
    ##########################################

    with open(f"{cfg_dir}/cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["scene"] = scene
    cfg["sample_idx"] = sample_idx
    cfg["cfg_dir"] = cfg_dir

    encoding = cfg["encoding"]["type"]
    n_levels = cfg["encoding"]["n_levels"]
    n_features_per_level = cfg["encoding"]["n_features_per_level"]
    base_resolution = cfg["encoding"]["base_resolution"]

    activation = cfg["mlp"]["activation"]
    use_single_mlp = cfg["mlp"]["use_single"]
    use_viewdirs = cfg["mlp"]["use_viewdirs"]
    n_neurons = cfg["mlp"]["n_neurons"]
    n_hidden_layers = cfg["mlp"]["n_hidden_layers"]

    run_name = \
        f"{scene}_search_perm" + \
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
