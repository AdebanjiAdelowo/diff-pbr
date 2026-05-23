"""Visualise the recovered material maps.

Loads the trained MaterialMaps checkpoint (or runs from the current
in-memory state after train.py) and saves PNGs for each of the four
material channels.

Usage
-----
    python visualize.py                          # after running train.py
    python visualize.py --ckpt results/mat.pth  # explicit checkpoint
"""

import argparse
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pbr.material import MaterialMaps


def sample_full_texture(mat: MaterialMaps, tex_h: int, tex_w: int,
                        device: torch.device):
    """Sample every material map over a full UV grid.

    Returns dict of (H, W, C) float32 numpy arrays.
    """
    u = torch.linspace(0, 1, tex_w, device=device)
    v = torch.linspace(0, 1, tex_h, device=device)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    uv_flat = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)   # (N,2)

    # Dummy TBN — identity (we only care about texture values here)
    N = uv_flat.shape[0]
    n_geo = torch.zeros(N, 3, device=device); n_geo[:, 2] = 1.0
    tan   = torch.zeros(N, 3, device=device); tan[:,  0] = 1.0
    bit   = torch.zeros(N, 3, device=device); bit[:,  1] = 1.0

    with torch.no_grad():
        albedo, n_ws, roughness, metallic = mat.sample(uv_flat, n_geo, tan, bit)

    def _arr(t, c): return t.cpu().float().numpy().reshape(tex_h, tex_w, c)
    return {
        "albedo":     _arr(albedo,    3),
        "roughness":  _arr(roughness, 1),
        "metallic":   _arr(metallic,  1),
        "normal_ws":  _arr(n_ws,      3),
    }


def gamma_correct(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img**(1/2.4) - 0.055)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default="results/mat.pth")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tex_res", type=int, default=64)
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

    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        # Infer tex resolution from checkpoint
        inferred_h, inferred_w = sd["log_albedo"].shape[2:]
        mat = MaterialMaps(tex_h=inferred_h, tex_w=inferred_w).to(device)
        mat.load_state_dict(sd)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"No checkpoint found at {args.ckpt}; visualising default init.")
        mat = MaterialMaps(tex_h=args.tex_res, tex_w=args.tex_res).to(device)
    mat.eval()

    maps = sample_full_texture(mat, args.tex_res, args.tex_res, device)

    fig = plt.figure(figsize=(12, 3.5), facecolor="#1a1a1a")
    gs  = gridspec.GridSpec(1, 4, hspace=0.04, wspace=0.06,
                            left=0.03, right=0.97, top=0.88, bottom=0.04)

    titles   = ["Albedo (sRGB)", "Normal Map", "Roughness", "Metallic"]
    channels = ["albedo",        "normal_ws",  "roughness", "metallic"]

    for j, (title, key) in enumerate(zip(titles, channels)):
        ax = fig.add_subplot(gs[0, j])
        arr = maps[key]
        if key == "albedo":
            arr = gamma_correct(np.clip(arr, 0, 1))
            img = (arr * 255).astype(np.uint8)
        elif key == "normal_ws":
            img = ((arr * 0.5 + 0.5) * 255).astype(np.uint8)
        else:
            img = (np.clip(arr[..., 0], 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, cmap="grey" if arr.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle("Recovered Material Maps", color="white", fontsize=13, y=0.97)
    out_path = os.path.join(args.out_dir, "material_maps.png")
    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Also save individual PNGs
    for key, arr in maps.items():
        if key == "albedo":
            arr_out = (gamma_correct(np.clip(arr, 0, 1)) * 255).astype(np.uint8)
        elif key == "normal_ws":
            arr_out = ((arr * 0.5 + 0.5) * 255).astype(np.uint8)
        else:
            arr_out = (np.clip(arr[..., 0], 0, 1) * 255).astype(np.uint8)
        plt.imsave(os.path.join(args.out_dir, f"{key}.png"), arr_out)
    print("Individual maps saved.")


if __name__ == "__main__":
    main()
