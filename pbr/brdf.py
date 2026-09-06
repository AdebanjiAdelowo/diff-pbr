"""Cook-Torrance microfacet BRDF.

Model
-----
    f_r = f_diffuse + f_specular

    f_diffuse  = (1 − F) · (1 − metallic) · albedo / π
    f_specular = D(h) · F(v,h) · G(l,v,h) / (4 · (n·l) · (n·v))

Components
----------
    D  — GGX / Trowbridge-Reitz normal distribution
    F  — Schlick Fresnel approximation
    G  — Smith height-correlated masking-shadowing

All operations are differentiable (pure PyTorch).

References
----------
- Walter et al. 2007 "Microfacet Models for Refraction"
- Karis 2013  "Real Shading in Unreal Engine 4"  (Epic GDC notes)
"""

import math
import torch
import torch.nn.functional as F


def _dot_clamped(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise dot product clamped to [0, 1]."""
    return (a * b).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)


def ggx_ndf(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """GGX normal distribution D(h).

    n_dot_h : (..., 1)  cosine of half-vector with normal
    alpha   : (..., 1)  roughness² (perceptual roughness squared)
    returns : (..., 1)
    """
    a2 = alpha * alpha
    denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom * denom + 1e-7)


def schlick_fresnel(f0: torch.Tensor, v_dot_h: torch.Tensor) -> torch.Tensor:
    """Schlick Fresnel F(v, h).

    f0      : (..., 3)  reflectance at normal incidence
    v_dot_h : (..., 1)  cosine of view direction with half-vector
    returns : (..., 3)
    """
    return f0 + (1.0 - f0) * (1.0 - v_dot_h) ** 5


def smith_ggx_g(n_dot_l: torch.Tensor, n_dot_v: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
    """Smith height-correlated G (joint masking-shadowing function).

    n_dot_l, n_dot_v : (..., 1)
    alpha            : (..., 1)
    returns          : (..., 1)
    """
    a2 = alpha * alpha

    def _lambda(cos_theta):
        cos2 = cos_theta * cos_theta
        tan2 = (1.0 - cos2) / (cos2 + 1e-7)
        return (-1.0 + (1.0 + a2 * tan2).sqrt()) * 0.5

    return 1.0 / (1.0 + _lambda(n_dot_l) + _lambda(n_dot_v) + 1e-7)


def cook_torrance(
    n:        torch.Tensor,   # (..., 3)  world-space normal
    l:        torch.Tensor,   # (..., 3)  unit direction to light (per-pixel)
    v:        torch.Tensor,   # (..., 3)  unit direction to camera (per-pixel)
    albedo:   torch.Tensor,   # (..., 3)  linear RGB in [0,1]
    roughness: torch.Tensor,  # (..., 1)  perceptual roughness in [0,1]
    metallic:  torch.Tensor,  # (..., 1)  metalness in [0,1]
    light_rgb: torch.Tensor,  # (..., 3)  incident radiance
) -> torch.Tensor:            # (..., 3)  outgoing radiance
    """Evaluate the Cook-Torrance BRDF at each fragment.

    Perceptual roughness is squared before use so that artists get a
    more linear feel — this follows the Epic/Karis convention.
    """
    alpha  = (roughness * roughness).clamp(min=0.001)
    h      = F.normalize(l + v, dim=-1)             # half vector

    n_dot_l = _dot_clamped(n, l)                    # (..., 1)
    n_dot_v = _dot_clamped(n, v).clamp(min=1e-4)
    n_dot_h = _dot_clamped(n, h)
    v_dot_h = _dot_clamped(v, h)

    # F0: dielectrics → 0.04, metals → albedo
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic   # (..., 3)

    D   = ggx_ndf(n_dot_h, alpha)                         # (..., 1)
    Fr  = schlick_fresnel(f0, v_dot_h)                    # (..., 3)
    G   = smith_ggx_g(n_dot_l, n_dot_v, alpha)            # (..., 1)

    specular = D * Fr * G / (4.0 * n_dot_l * n_dot_v + 1e-7)  # (..., 3)

    k_diffuse = (1.0 - Fr) * (1.0 - metallic)
    diffuse   = k_diffuse * albedo / math.pi

    return (diffuse + specular) * n_dot_l * light_rgb
