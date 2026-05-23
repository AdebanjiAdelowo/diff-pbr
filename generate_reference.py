"""Render ground-truth reference images for the diff-pbr inverse problem.

A ground-truth MaterialMaps is constructed with known parameters, then
rendered from each camera view.  Images are saved as PNG and also
stored as float32 .npy tensors for the training script.

Usage
-----
    python generate_reference.py                 # default 8 views, 64-px tex
    python generate_reference.py --res 128 --views 12
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbr.mesh import uv_sphere
from pbr.camera import hemisphere_cameras, default_lights
from pbr.rasterize import rasterize
from pbr.material import MaterialMaps
from pbr.render import render


def gamma_correct(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img**(1/2.4) - 0.055)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--views",   type=int, default=8)
    p.add_argument("--img_res", type=int, default=128, help="image resolution")
    p.add_argument("--tex_res", type=int, default=64,  help="texture resolution")
    p.add_argument("--out_dir", default="data/reference")
    p.add_argument("--device",  default="")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    mesh    = uv_sphere(rings=32, sectors=64)
    cameras = hemisphere_cameras(n=args.views, img_w=args.img_res, img_h=args.img_res)
    lights  = default_lights()

    # Ground-truth material: orange-ish dielectric, medium roughness
    gt_material = MaterialMaps(
        tex_h=args.tex_res,
        tex_w=args.tex_res,
        init_albedo=(0.72, 0.35, 0.12),
        init_roughness=0.45,
        init_metallic=0.02,
    ).to(device)
    gt_material.eval()

    # Add some spatial variation to the ground-truth albedo texture
    # (a sinusoidal pattern so there is something non-trivial to recover)
    with torch.no_grad():
        H, W = args.tex_res, args.tex_res
        u = torch.linspace(-math.pi, math.pi, W, device=device)
        v = torch.linspace(-math.pi, math.pi, H, device=device)
        gu, gv = torch.meshgrid(u, v, indexing="xy")
        pattern = 0.15 * torch.sin(gu * 3) * torch.cos(gv * 2)   # (H,W)
        gt_material.log_albedo.data[0, 0] += pattern
        gt_material.log_albedo.data[0, 1] += pattern * 0.5

        # A specular metallic band around the equator
        band = torch.exp(-((gv) ** 2) / 0.5)   # (H,W) Gaussian in v
        import math as _math
        metal_logit = _math.log(0.85 / 0.15)
        gt_material.log_metallic.data[0, 0] = band * metal_logit

    # Pre-rasterize all views
    print("Rasterizing views …")
    geo_buffers = []
    for i, cam in enumerate(cameras):
        geo_buffers.append(rasterize(mesh, cam))
        print(f"  View {i+1}/{len(cameras)} done")

    np.save(os.path.join(args.out_dir, "geo_buffers.npy"),
            geo_buffers, allow_pickle=True)

    # Render reference images
    print("Rendering reference images …")
    ref_images = []
    for i, (cam, geo) in enumerate(zip(cameras, geo_buffers)):
        with torch.no_grad():
            img_t = render(geo, cam, lights, gt_material, device)
        img_np = img_t.cpu().float().numpy()
        ref_images.append(img_np)
        # Save float tensor
        np.save(os.path.join(args.out_dir, f"ref_{i:02d}.npy"), img_np)
        # Save PNG (gamma-corrected)
        img_u8 = (gamma_correct(np.clip(img_np, 0, 1)) * 255).astype(np.uint8)
        plt.imsave(os.path.join(args.out_dir, f"ref_{i:02d}.png"), img_u8)
        print(f"  Saved ref_{i:02d}.png")

    # Save camera count for the training script
    np.save(os.path.join(args.out_dir, "n_views.npy"), np.array(args.views))
    print(f"\nReference data written to {args.out_dir}/")
    print("Run  python train.py  to start the inverse-rendering optimisation.")


import math
if __name__ == "__main__":
    main()
