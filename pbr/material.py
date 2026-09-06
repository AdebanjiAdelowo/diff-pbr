"""Learnable material maps (differentiable texture atlases).

Four learnable textures, each stored in log/logit space for
unconstrained gradient flow:

    albedo      (3-channel)  → sigmoid → [0,1]³   (linear sRGB)
    normal_delta(2-channel)  → tanh    → [-1,1]²  (tangent-space XY perturbation)
    roughness   (1-channel)  → sigmoid → [0,1]    (perceptual roughness)
    metallic    (1-channel)  → sigmoid → [0,1]    (metalness)

Texture sampling uses a hand-rolled bilinear interpolation built on
torch.gather (see MaterialMaps._sample), not F.grid_sample: grid_sample's
backward pass is not implemented on the PyTorch MPS backend, so this
gather-based sampler is used instead to keep gradients working on CPU,
CUDA, and MPS alike. UV coords are expected in [0,1] and are mapped
directly to pixel coordinates [0, W-1] / [0, H-1].

The normal delta is converted to a full tangent-space normal via:
    n_ts = normalise(Δx, Δy, 1.0)
and then rotated to world space using the TBN matrix from the rasterizer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # used in sample() for F.normalize
from typing import Tuple


class MaterialMaps(nn.Module):
    """Four learnable texture maps for PBR material optimisation.

    Parameters
    ----------
    tex_h, tex_w : texture resolution (default 64×64)
    init_albedo  : (3,) initial linear-sRGB albedo
    init_roughness : float initial perceptual roughness
    init_metallic  : float initial metalness
    """

    def __init__(
        self,
        tex_h: int = 64,
        tex_w: int = 64,
        init_albedo: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        init_roughness: float = 0.5,
        init_metallic:  float = 0.0,
    ):
        super().__init__()
        self.tex_h = tex_h
        self.tex_w = tex_w

        def _inv_sigmoid(x):
            x = max(min(x, 1.0 - 1e-6), 1e-6)
            return math.log(x / (1.0 - x))

        # Albedo: (1, 3, H, W)
        a_init = torch.tensor(init_albedo).float()
        a_logit = torch.stack([
            torch.full((tex_h, tex_w), _inv_sigmoid(c)) for c in a_init
        ]).unsqueeze(0)
        self.log_albedo = nn.Parameter(a_logit)

        # Normal delta: (1, 2, H, W) — initialised to zero (flat normals)
        self.normal_delta = nn.Parameter(torch.zeros(1, 2, tex_h, tex_w))

        # Roughness: (1, 1, H, W)
        r_logit = _inv_sigmoid(init_roughness)
        self.log_roughness = nn.Parameter(
            torch.full((1, 1, tex_h, tex_w), r_logit))

        # Metallic: (1, 1, H, W)
        m_logit = _inv_sigmoid(init_metallic)
        self.log_metallic = nn.Parameter(
            torch.full((1, 1, tex_h, tex_w), m_logit))

    def _sample(self, tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear sample using torch.gather — MPS-safe (no grid_sample backward).

        tex : (1, C, H, W)
        uv  : (N, 2)  in [0,1]

        Returns (N, C).
        """
        _, C, H, W = tex.shape
        N = uv.shape[0]

        # Map UV [0,1] to pixel coordinates [0, W-1] / [0, H-1]
        px = uv[:, 0] * (W - 1)
        py = uv[:, 1] * (H - 1)

        x0 = px.long().clamp(0, W - 2)
        y0 = py.long().clamp(0, H - 2)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        wx = (px - x0.float()).unsqueeze(-1)   # (N, 1)
        wy = (py - y0.float()).unsqueeze(-1)   # (N, 1)

        flat = tex[0].view(C, H * W)           # (C, H*W)

        def _fetch(xi, yi):
            idx = (yi * W + xi).unsqueeze(0).expand(C, N)  # (C, N)
            return flat.gather(1, idx).T                    # (N, C)

        return (_fetch(x0, y0) * (1 - wx) * (1 - wy)
              + _fetch(x1, y0) *      wx  * (1 - wy)
              + _fetch(x0, y1) * (1 - wx) *      wy
              + _fetch(x1, y1) *      wx  *      wy)

    def sample(
        self,
        uv: torch.Tensor,          # (N, 2)
        pixel_normal:  torch.Tensor,   # (N, 3) world-space geometric normal
        pixel_tangent: torch.Tensor,   # (N, 3) world-space tangent
        pixel_bitangent: torch.Tensor, # (N, 3) world-space bitangent
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample all maps at UV and return per-pixel material values.

        Returns
        -------
        albedo    : (N, 3)
        normal_ws : (N, 3)  world-space shading normal (after normal map)
        roughness : (N, 1)
        metallic  : (N, 1)
        """
        albedo    = torch.sigmoid(self._sample(self.log_albedo,    uv))
        n_delta   = torch.tanh(  self._sample(self.normal_delta,  uv))   # (N,2)
        roughness = torch.sigmoid(self._sample(self.log_roughness, uv))
        metallic  = torch.sigmoid(self._sample(self.log_metallic,  uv))

        # Build tangent-space normal and rotate to world space
        # n_ts = normalise(Δx, Δy, 1)
        nx = n_delta[:, 0:1]
        ny = n_delta[:, 1:2]
        nz = torch.ones_like(nx)
        n_ts = F.normalize(torch.cat([nx, ny, nz], dim=-1), dim=-1)  # (N,3)

        # TBN columns: T, B, N  → world-space normal = T·n_ts.x + B·n_ts.y + N·n_ts.z
        normal_ws = (pixel_tangent   * n_ts[:, 0:1]
                   + pixel_bitangent * n_ts[:, 1:2]
                   + pixel_normal    * n_ts[:, 2:3])
        normal_ws = F.normalize(normal_ws, dim=-1)

        return albedo, normal_ws, roughness, metallic
