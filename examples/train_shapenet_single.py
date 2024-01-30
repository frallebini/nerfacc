import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import wandb
import yaml

from lpips import LPIPS
from pathlib import Path
from tqdm import tqdm

from datasets.shapenet import ShapeNetLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from radiance_fields.ngp_single_mlp import NGPRadianceFieldSingleMlp
from utils import render_image_with_occgrid, render_image_with_occgrid_test

device = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg-path",
    type=str,
    help="the path to the .yml configuration file",
)
parser.add_argument(
    "--id",
    type=str,
    help="class_id/obj_id",
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

log2_target_sample_batch_size = cfg["log2_target_sample_batch_size"]
lr = cfg["lr"]

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

run_name = \
    f"{args.id}_{encoding}_{activation.lower()}" + \
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

# training parameters
max_steps = 1500
render_n_samples = 1024
target_sample_batch_size = 1 << log2_target_sample_batch_size
# scene parameters
aabb = torch.tensor([-0.7, -0.7, -0.7, 0.7, 0.7, 0.7], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = (
    (aabb[3:] - aabb[:3]).max() * math.sqrt(3) / render_n_samples
).item()
alpha_thre = 0.0
cone_angle = 0.0

train_dataset = ShapeNetLoader(
    subject_id=args.id,
    root_fp=data_root,
    split="train",
    color_bkgd_aug="random",
    num_rays=target_sample_batch_size // render_n_samples,
    device=device
)
test_dataset = ShapeNetLoader(
    subject_id=args.id,
    root_fp=data_root,
    split="train",
    num_rays=None,
    device=device
)

estimator = OccGridEstimator(
    roi_aabb=aabb, 
    resolution=grid_resolution, 
    levels=grid_nlvl
).to(device)

# setup the radiance field we want to train
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
    ).to(device)
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
    ).to(device)

sd_dir = Path(
    f"ckpts/{encoding}_{activation.lower()}" + \
    f"{'_single' if use_single_mlp else ''}" + \
    f"{'_viewdir' if use_viewdirs else ''}" + \
    f"_{n_levels}_{n_features_per_level}_{base_resolution}"
)
sd_dir.mkdir(parents=True, exist_ok=True)
sd_path = sd_dir / "shapenet" / f"init_{args.model_name}.pt"

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
optimizer = torch.optim.Adam(
    radiance_field.parameters(), 
    lr=lr, 
    eps=1e-15
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
step = 0
failed_steps = 0
pbar = tqdm(total=max_steps, desc="train step")
tic = time.time()

while step < max_steps:
    for i in range(len(train_dataset)):
        radiance_field.train()
        estimator.train()

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
        )

        # render
        rgb, acc, _, n_rendering_samples = render_image_with_occgrid(
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
        if n_rendering_samples == 0:
            tqdm.write("0 rendering samples")
            continue

        # dynamic batch size for rays to keep sample batch size constant
        num_rays = len(pixels)
        num_rays = int(
            num_rays
            * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)
        
        alive_ray_mask = acc.squeeze(-1) > 0

        # compute loss
        loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

        optimizer.zero_grad()
        # do not unscale it because we are using Adam
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            wandb.log({
                "train/loss": loss,
                "train/psnr": psnr,
            }, step=step)
            
        step += 1
        pbar.update(1)

elapsed_time = time.time() - tic
wandb.log({"train/elapsed_time": elapsed_time})

pbar.close()

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
            tqdm.write(f"skipped test sample {i}")
            continue

        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())
        
        if i % 6 == 0:
            rendered = rgb.cpu().numpy()
            gt = f"{data_root}/{args.id}/train/{i:02}.png"
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
class_id, object_id = args.id.split("/")
(sd_dir / "shapenet" /class_id).mkdir(exist_ok=True)
torch.save(sd, sd_dir / "shapenet" / f"{args.id}_{args.model_name}.pt")
