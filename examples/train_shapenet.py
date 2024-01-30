import gc
import math
import torch
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm

from datasets.shapenet import CLASS_DICT, TO_SKIP, ShapeNetLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from utils import render_image_with_occgrid


def train_nerf(data_root: str, out_root: Path, full_id: str) -> None:

    device = "cuda:0"

    # training parameters
    max_steps = 1500
    render_n_samples = 1024
    lr = 1e-2
    target_sample_batch_size = 1 << 18
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
        subject_id=full_id,
        root_fp=data_root,
        split="train",
        color_bkgd_aug="random",
        num_rays=target_sample_batch_size // render_n_samples,
        device=device
    )
    
    estimator = OccGridEstimator(
        roi_aabb=aabb, 
        resolution=grid_resolution, 
        levels=grid_nlvl
    ).to(device)
    
    # setup the radiance field we want to train
    radiance_field = NGPRadianceField(
        aabb=estimator.aabbs[-1],
        encoding_type="cuda"
    ).to(device)

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

    # training
    step = 0
    failed_steps = 0

    while step < max_steps:
        for i in range(len(train_dataset)):
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
                occ_eval_fn=occ_eval_fn
            )

            # render
            try:
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
            except RuntimeError:  # OOM
                with open(f"{out_root}/failed.txt", "a") as f:
                    f.write(f"{full_id}\n")
                gc.collect()
                torch.cuda.empty_cache()
                return
            
            if n_rendering_samples == 0:
                failed_steps += 1
                if failed_steps == 10:
                    with open(f"{out_root}/failed.txt", "a") as f:
                        f.write(f"{full_id}\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                    return
                else:
                    continue

            # dynamic batch size for rays to keep sample batch size constant
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            try:
                train_dataset.update_num_rays(num_rays)
            except RuntimeError:  # OOM
                with open(f"{out_root}/failed.txt", "a") as f:
                    f.write(f"{full_id}\n")
                gc.collect()
                torch.cuda.empty_cache()
                return
                
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam
            try:
                grad_scaler.scale(loss).backward()
            except RuntimeError:  # matmul error
                with open(f"{out_root}/failed.txt", "a") as f:
                    f.write(f"{full_id}\n")
                gc.collect()
                torch.cuda.empty_cache()
                return
            optimizer.step()
            scheduler.step()
            
            step += 1
    
    # save model weights
    sd = {
        "radiance_field": radiance_field.state_dict(),
        "estimator": estimator.state_dict()
    }
    class_id = CLASS_DICT[full_id.split("/")[0]]
    sd["class_id"] = class_id
    torch.save(sd, f"{out_root / full_id}.pt")

    gc.collect()
    torch.cuda.empty_cache()


def train_dset() -> None:
    data_root = "/media/data7/fballerini/datasets/shapenet_render"
    
    out_root = Path("/media/data2/fballerini/datasets/shapenet_render_instant")
    out_root.mkdir(parents=True, exist_ok=True)
    
    failed_file = out_root / "failed.txt"
    if failed_file.exists():
        failed_ids = []
        with open(failed_file, "r") as f:
            for line in f:
                failed_ids.append(line[:-1])

    class_paths = sorted(dir for dir in Path(data_root).iterdir() if dir.stem in CLASS_DICT.keys())
    obj_paths = [path for c_path in class_paths for path in sorted(c_path.iterdir())]
    
    for obj_path in tqdm(obj_paths):
        class_id = obj_path.parent.stem
        obj_id = obj_path.stem
        full_id = f"{class_id}/{obj_id}"
        
        if full_id in TO_SKIP:
            continue
        
        if full_id in failed_ids:
            continue
        
        out_file = out_root / class_id / f"{obj_id}.pt"
        if out_file.exists():
            continue
        
        out_path = out_root / class_id
        out_path.mkdir(exist_ok=True)
        
        train_nerf(data_root, out_root, full_id)


if __name__ == "__main__":
    train_dset()
