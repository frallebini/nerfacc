"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import warnings
warnings.simplefilter("ignore", UserWarning)

import os
os.environ["WANDB_SILENT"] = "true"

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from lpips import LPIPS
from pathlib import Path

from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(Path.cwd() / "data/360_v2"),
    default=str(Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--encoding",
    type=str,
    default="combo",
    choices=["combo", "cuda", "torch"],
    help="which type of multi-resolution hash grid encoding to use",
)
args = parser.parse_args()

run_name = f"{args.scene}_{args.encoding}_hash_A_mlp_B"
wandb.init(
        entity="frallebini",
        project="nerfacc",
        name=run_name,
        config=args,
)

device = "cuda:0"
# set_random_seed(42)

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
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
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
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train
if args.encoding == "cuda":
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], use_cuda_encoding=True)
elif args.encoding == "torch":
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], use_torch_encoding=True)
    #################### SAVE INIT #####################
    # ckpt = {
    #     "radiance_field": radiance_field.state_dict(),
    #     "estimator": estimator.state_dict(),
    # }
    # torch.save(ckpt, "ckpts/init_B.pt")
    ####################################################
else:
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1])
radiance_field = radiance_field.to(device)

############################## LOAD INIT ####################################
sd_A = torch.load("ckpts/init_A.pt")
sd_B = torch.load("ckpts/init_B.pt")
sd_A["radiance_field"]["mlp_base.1.params"] = sd_B["radiance_field"]["mlp_base.1.params"]  # density mlp
sd_A["radiance_field"]["mlp_head.params"] = sd_B["radiance_field"]["mlp_head.params"]  # color mlp
estimator.load_state_dict(sd_B["estimator"])
radiance_field.load_state_dict(sd_A["radiance_field"])
#############################################################################

grad_scaler = torch.cuda.amp.GradScaler(2**10)
optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
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
for step in tqdm.tqdm(range(max_steps + 1), "train step"):
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
    scheduler.step()

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
            for i in tqdm.tqdm(range(len(test_dataset)), "test sample"):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

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
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                if i % 10 == 0:
                    rendered = (rgb.cpu().numpy() * 255).astype(np.uint8)
                    gt = f"{args.data_root}/{args.scene}/test/r_{i}.png"
                    wandb.log({f"test/{i}": [wandb.Image(gt), wandb.Image(rendered)]})
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        wandb.log({
            "test/psnr_avg": psnr_avg,
            "test/lpips_avg": lpips_avg,
        })
        
        Path("ckpts").mkdir(exist_ok=True)
        ckpt = {
            "radiance_field": radiance_field.state_dict(),
            "estimator": estimator.state_dict(),
        }
        torch.save(ckpt, f"ckpts/{run_name}.pt")
