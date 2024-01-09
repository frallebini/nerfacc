"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import wandb
import yaml

from lpips import LPIPS
from pathlib import Path
from tqdm import tqdm

from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from radiance_fields.ngp_single_mlp import NGPRadianceFieldSingleMlp
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
)

device = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg-path",
    type=str,
    help="the path to the .yml configuration file",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="A",
    choices=["A", "B"],
    help="which alphanumeric label to use to refer to the model",
)
parser.add_argument(
    "--save-init", 
    action="store_true",
    help="whether to save the initial model weights"
)
parser.add_argument(
    "--load-init", 
    action="store_true",
    help="whether to load precomputed initial weights"
)
args = parser.parse_args()

with open(args.cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
data_root = cfg["data_root"]
train_split = cfg["train_split"]

log2_target_sample_batch_size = cfg["log2_target_sample_batch_size"]
lr = cfg["lr"]

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
    f"{args.scene}_{encoding}_{activation.lower()}" + \
    f"{'_single' if use_single_mlp else ''}" + \
    f"{'_viewdir' if use_viewdirs else ''}" + \
    f"_{n_levels}_{n_features_per_level}_{base_resolution}" + \
    f"_{args.model_name}"
wandb.init(
    entity="frallebini",
    project="nerfacc",
    name=run_name,
    config={**cfg, **vars(args)},
)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20_000
    init_batch_size = 1024
    target_sample_batch_size = 1 << log2_target_sample_batch_size
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root,
    split=train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train
if use_single_mlp:
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
    ).to(device)
else:
    radiance_field = NGPRadianceField(
        aabb=estimator.aabbs[-1],
        use_viewdirs=use_viewdirs,
        base_resolution=base_resolution,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        encoding_type=encoding,
        mlp_activation=activation,
    ).to(device)

sd_dir = Path(
    f"ckpts/{encoding}_{activation.lower()}" + \
    f"{'_single' if use_single_mlp else ''}" + \
    f"{'_viewdir' if use_viewdirs else ''}" + \
    f"_{n_levels}_{n_features_per_level}_{base_resolution}"
)
sd_dir.mkdir(parents=True, exist_ok=True)
sd_path = sd_dir / f"init_{args.model_name}.pt"

if args.save_init:
    sd = {
        "radiance_field": radiance_field.state_dict(),
        "estimator": estimator.state_dict(),
    }
    torch.save(sd, sd_path)
    
if args.load_init:
    sd = torch.load(sd_path)
    estimator.load_state_dict(sd["estimator"])
    radiance_field.load_state_dict(sd["radiance_field"])

grad_scaler = torch.cuda.amp.GradScaler(2**10)

if use_single_mlp:
    optimizer = torch.optim.AdamW(radiance_field.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps)
else:
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=lr, eps=1e-15, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in tqdm(range(max_steps + 1), "train step"):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
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
    if n_rendering_samples == 0:
        tqdm.write("0 rendering samples")
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam
    grad_scaler.scale(loss).backward()
    optimizer.step()
    try:
        scheduler.step()
    except ValueError as e:
        tqdm.write(str(e))

    if step % 100 == 0:
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        wandb.log({
            "train/loss": loss,
            "train/psnr": psnr,
        }, step=step)

    if step > 0 and step % max_steps == 0:
        elapsed_time = time.time() - tic
        wandb.log({"train/elapsed_time": elapsed_time})

        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), "test sample"):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                try:
                    # rendering
                    rgb, acc, depth, _ = render_image_with_occgrid_test(
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
                    tqdm.write(f"skipped test sample {i}")
                    continue

                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                
                if i % 10 == 0:
                    rendered = (rgb.cpu().numpy() * 255).astype(np.uint8)
                    gt = f"{data_root}/{args.scene}/test/r_{i}.png"
                    wandb.log({f"test/{i}": [wandb.Image(gt), wandb.Image(rendered)]})
        
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        wandb.log({
            "test/psnr_avg": psnr_avg,
            "test/lpips_avg": lpips_avg,
        })

        sd = {
            "radiance_field": radiance_field.state_dict(),
            "estimator": estimator.state_dict(),
        }
        torch.save(sd, sd_dir / f"{args.scene}_{args.model_name}.pt")
