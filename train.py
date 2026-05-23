"""Inverse-rendering optimisation loop.

Optimises MaterialMaps parameters to minimise the L2 photometric loss
against the reference images produced by generate_reference.py.

Usage
-----
    python train.py                                   # defaults
    python train.py --steps 3000 --lr 1e-2 --tex_res 64
    python train.py --device mps
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbr.mesh import uv_sphere
from pbr.camera import hemisphere_cameras, default_lights
from pbr.rasterize import rasterize, GeometryBuffers
from pbr.material import MaterialMaps
from pbr.render import render


def gamma_correct(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img**(1/2.4) - 0.055)


def load_reference(data_dir: str, n_views: int, device: torch.device):
    refs = []
    for i in range(n_views):
        arr = np.load(os.path.join(data_dir, f"ref_{i:02d}.npy"))
        refs.append(torch.tensor(arr, device=device, dtype=torch.float32))
    return refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/reference")
    p.add_argument("--out_dir",  default="results")
    p.add_argument("--steps",    type=int,   default=2000)
    p.add_argument("--lr",       type=float, default=2e-2)
    p.add_argument("--tex_res",  type=int,   default=64)
    p.add_argument("--img_res",  type=int,   default=128)
    p.add_argument("--log_every",type=int,   default=100)
    p.add_argument("--device",   default="")
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

    # Load reference data
    n_views = int(np.load(os.path.join(args.data_dir, "n_views.npy")))
    refs    = load_reference(args.data_dir, n_views, device)

    geo_buffers = list(np.load(
        os.path.join(args.data_dir, "geo_buffers.npy"), allow_pickle=True))

    cameras = hemisphere_cameras(n=n_views, img_w=args.img_res, img_h=args.img_res)
    lights  = default_lights()

    # Initialise material with a neutral grey guess
    material = MaterialMaps(
        tex_h=args.tex_res,
        tex_w=args.tex_res,
        init_albedo=(0.5, 0.5, 0.5),
        init_roughness=0.7,
        init_metallic=0.0,
    ).to(device)
    material.train()

    optimizer = torch.optim.Adam(material.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    losses = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # Pick a random view each step
        vi = np.random.randint(n_views)
        cam, geo, ref = cameras[vi], geo_buffers[vi], refs[vi]

        optimizer.zero_grad()
        pred = render(geo, cam, lights, material, device)

        mask = torch.tensor(geo.pixel_mask, device=device)  # (H,W)
        loss = F.mse_loss(pred[mask], ref[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(material.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-args.log_every:])
            print(f"  step {step:5d}/{args.steps}  loss={avg_loss:.5f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}"
                  f"  {elapsed:.1f}s elapsed")

    # Save loss curve
    plt.figure(figsize=(7, 3))
    plt.semilogy(losses)
    plt.xlabel("Step"); plt.ylabel("MSE loss"); plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss.png"), dpi=120)
    plt.close()

    # Render final predictions for all views
    material.eval()
    print("\nRendering final predictions …")
    for vi, (cam, geo, ref) in enumerate(zip(cameras, geo_buffers, refs)):
        with torch.no_grad():
            pred = render(geo, cam, lights, material, device)

        pred_np = gamma_correct(np.clip(pred.cpu().float().numpy(), 0, 1))
        ref_np  = gamma_correct(np.clip(ref.cpu().float().numpy(),  0, 1))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow((ref_np  * 255).astype(np.uint8)); axes[0].set_title("Reference")
        axes[1].imshow((pred_np * 255).astype(np.uint8)); axes[1].set_title("Recovered")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"cmp_{vi:02d}.png"), dpi=120)
        plt.close()

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, "mat.pth")
    torch.save(material.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    print(f"\nResults saved to {args.out_dir}/")
    print("Run  python visualize.py  to inspect the recovered material maps.")


if __name__ == "__main__":
    main()
