"""Differentiable rendering pass.

Given pre-computed geometry buffers (UV, normal, position) and a
MaterialMaps module, evaluates the Cook-Torrance BRDF at each surface
pixel and returns a (H, W, 3) linear-sRGB image.

The rasterization step (geometry buffers) is non-differentiable but
fixed; all texture sampling and shading operations below are
differentiable with respect to material parameters.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .brdf import cook_torrance
from .camera import Camera, PointLight
from .material import MaterialMaps
from .rasterize import GeometryBuffers
from typing import List


def render(
    geo:      GeometryBuffers,
    cam:      Camera,
    lights:   List[PointLight],
    material: MaterialMaps,
    device:   torch.device,
) -> torch.Tensor:            # (H, W, 3) linear radiance, requires_grad
    """Shade the surface patch.

    Non-masked pixels are returned as 0 (black background).
    """
    H, W = cam.height, cam.width
    mask = geo.pixel_mask     # (H, W)

    if not mask.any():
        return torch.zeros(H, W, 3, device=device)

    # Flatten masked pixels
    my, mx = np.where(mask)                          # (N,), (N,)
    uv   = torch.tensor(geo.pixel_uv[my, mx],        device=device)  # (N,2)
    pos  = torch.tensor(geo.pixel_pos[my, mx],       device=device)  # (N,3)
    n_geo = torch.tensor(geo.pixel_normal[my, mx],   device=device)  # (N,3)
    tan  = torch.tensor(geo.pixel_tangent[my, mx],   device=device)  # (N,3)
    bit  = torch.tensor(geo.pixel_bitangent[my, mx], device=device)  # (N,3)

    # View direction (camera → fragment, normalised)
    cam_pos = torch.tensor(cam.position, device=device).float()
    v = F.normalize(cam_pos.unsqueeze(0) - pos, dim=-1)               # (N,3)

    # Sample material maps
    albedo, n_shading, roughness, metallic = material.sample(
        uv, n_geo, tan, bit)

    # Accumulate contribution from each light
    out = torch.zeros(len(my), 3, device=device)
    for light in lights:
        l_pos   = torch.tensor(light.position, device=device).float()
        l_color = torch.tensor(light.color,    device=device).float()

        l_dir = F.normalize(l_pos.unsqueeze(0) - pos, dim=-1)         # (N,3)

        # Inverse-square attenuation
        dist2 = ((l_pos.unsqueeze(0) - pos) ** 2).sum(-1, keepdim=True).clamp(min=0.01)
        li = l_color.unsqueeze(0) / dist2                              # (N,3)

        out += cook_torrance(n_shading, l_dir, v, albedo, roughness, metallic, li)

    # Scatter back to full image (background stays 0)
    image = torch.zeros(H * W, 3, device=device)
    idx   = torch.tensor(my * W + mx, device=device, dtype=torch.long)
    image.scatter_(0, idx.unsqueeze(-1).expand(-1, 3), out)
    return image.reshape(H, W, 3)
