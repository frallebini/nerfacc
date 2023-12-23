import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets.nerf_synthetic import SubjectLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from radiance_fields.ngp_single_mlp import NGPRadianceFieldSingleMlp

from torch import Tensor
from tqdm import tqdm
from typing import Any, Dict, List
from utils import render_image_with_occgrid_test


def get_state_dict(cfg: Dict[str, Any]) -> Dict[str, Tensor]:
    sd_A = torch.load(cfg["path_A"])
    sd_B = torch.load(cfg["path_B"])
    if "single" in cfg["run_name"]:
        sd_A["radiance_field"]["mlp.params"] = sd_B["radiance_field"]["mlp.params"]
    else:
        sd_A["radiance_field"]["mlp_base.1.params"] = sd_B["radiance_field"]["mlp_base.1.params"]  # density mlp
        sd_A["radiance_field"]["mlp_head.params"] = sd_B["radiance_field"]["mlp_head.params"]  # color mlp

    return sd_A


def generate_binary_masks(bit_count: int) -> List[List[bool]]:
    # Returns the list of binary masks of length bit_count.
    # Example:
    # >>> generate_binary_masks(3)
    # [[F,F,F],[F,F,T],[F,T,F],[F,T,T],[T,F,F],[T,F,T],[T,T,F],[T,T,T]]
    binary_masks = []
    
    def genbin(n: int, bm=[]) -> None:
        if len(bm) == n:
            binary_masks.append(bm)
        else:
            genbin(n, bm + [False])
            genbin(n, bm + [True])

    genbin(bit_count)
    return binary_masks


def search_perm(cfg: Dict[str, Any]) -> None:
    # dataset parameters
    scene = "drums"
    data_root = "/media/data7/fballerini/datasets/nerf_synthetic"
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

    is_sine = "sine" in cfg["run_name"]
    is_single_mlp = "single" in cfg["run_name"]

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
    
    if is_single_mlp:
        radiance_field = NGPRadianceFieldSingleMlp(
            aabb=estimator.aabbs[-1], 
            encoding_type="torch",
            mlp_activation="Sine" if is_sine else "ReLU"
        ).cuda()
    else:
        radiance_field = NGPRadianceField(
            aabb=estimator.aabbs[-1], 
            encoding_type="torch",
            mlp_activation="Sine" if is_sine else "ReLU"
        ).cuda()


    psnr_best = 0
    # permute columns of tensor of shape (n, 2)
    perm_cols = lambda t: torch.index_select(t, dim=1, index=torch.LongTensor([1, 0]))

    for step, mask in enumerate(tqdm(generate_binary_masks(16))):
        # reset state dict
        sd = get_state_dict(cfg)

        # get grids
        if is_single_mlp:
            grids = [sd["radiance_field"][f"encoding.levels.{i}.embedding.weight"].cpu() for i in range(16)]
        else:
            grids = [sd["radiance_field"][f"mlp_base.0.levels.{i}.embedding.weight"].cpu() for i in range(16)]
        grids = np.array(grids, dtype="object")

        # permute grids
        selected_grids = grids[mask]
        swapped_grids = [perm_cols(grid) for grid in selected_grids]
        swapped_grids = np.array(swapped_grids, dtype="object")
        grids[mask] = swapped_grids

        # set grids
        for i in range(16):
            if is_single_mlp:
                sd["radiance_field"][f"encoding.levels.{i}.embedding.weight"] = grids[i].cuda()
            else:
                sd["radiance_field"][f"mlp_base.0.levels.{i}.embedding.weight"] = grids[i].cuda()
        estimator.load_state_dict(sd["estimator"])
        radiance_field.load_state_dict(sd["radiance_field"])

        # evaluation
        estimator.eval()
        radiance_field.eval()

        with torch.no_grad():
            data = test_dataset[cfg["sample_idx"]]
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
                tqdm.write(f"skipped binary mask {[int(b) for b in mask]}")
                continue

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            wandb.log({"test/psnr": psnr}, step=step)
        
        if psnr > psnr_best:
            psnr_best = psnr
            wandb.log({"test/psnr_best": psnr_best}, step=step)
            torch.save(sd, cfg["path_out"])


if __name__ == "__main__":
    scene = "chair"
    activation = "sine_single"

    cfg = {
        "run_name": f"{scene}_search_perm_{activation}",
        "sample_idx": 42,
        "path_A": f"ckpts/{scene}_torch_{activation}_A.pt",
        "path_B": f"ckpts/{scene}_torch_{activation}_B.pt",
        "path_out": f"ckpts/{scene}_torch_{activation}_perm.pt"
    }

    wandb.init(
        entity="frallebini",
        project="nerfacc",
        name=cfg["run_name"],
        config=cfg
    )
    
    search_perm(cfg)
