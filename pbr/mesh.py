"""UV sphere mesh with analytically correct TBN frames.

The sphere is parameterised by longitude φ ∈ [0, 2π) and
latitude  θ ∈ [0, π], giving:

    P(θ, φ) = (sin θ cos φ, cos θ, sin θ sin φ)   (unit sphere)

UV coordinates are the standard equirectangular mapping:
    u = φ / (2π),   v = θ / π

Tangent (∂P/∂u normalised) and bitangent (∂P/∂v normalised)
are derived analytically and stored per vertex.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Mesh:
    """Triangle mesh with per-vertex UV, normal, tangent, bitangent."""
    verts:     np.ndarray   # (V, 3)  float32 positions
    uvs:       np.ndarray   # (V, 2)  float32 UV coords in [0,1]
    normals:   np.ndarray   # (V, 3)  float32 outward unit normals
    tangents:  np.ndarray   # (V, 3)  float32 dP/du direction
    bitangents: np.ndarray  # (V, 3)  float32 dP/dv direction
    faces:     np.ndarray   # (F, 3)  int32  vertex indices


def uv_sphere(rings: int = 32, sectors: int = 64) -> Mesh:
    """Build a UV sphere of radius 1.

    Parameters
    ----------
    rings   : int  latitude subdivisions  (poles excluded → rings+1 lat lines)
    sectors : int  longitude subdivisions
    """
    verts, uvs, normals, tangents, bitangents = [], [], [], [], []

    for ri in range(rings + 1):
        theta = math.pi * ri / rings          # [0, π]
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        v_coord = ri / rings                  # [0, 1]

        for si in range(sectors + 1):
            phi = 2.0 * math.pi * si / sectors   # [0, 2π]
            sin_p = math.sin(phi)
            cos_p = math.cos(phi)
            u_coord = si / sectors               # [0, 1]

            # Position on unit sphere
            x = sin_t * cos_p
            y = cos_t
            z = sin_t * sin_p
            verts.append([x, y, z])
            uvs.append([u_coord, v_coord])

            # Normal = position on unit sphere
            normals.append([x, y, z])

            # Tangent = ∂P/∂u (normalised) = ∂P/∂φ · 2π
            # ∂P/∂φ = (-sin_t·sin_p, 0, sin_t·cos_p)
            tx = -sin_p
            ty = 0.0
            tz = cos_p
            # Already unit length (for sin_t ≠ 0)
            length = math.sqrt(tx*tx + ty*ty + tz*tz)
            if length > 1e-6:
                tx, ty, tz = tx/length, ty/length, tz/length
            else:
                tx, ty, tz = 1.0, 0.0, 0.0
            tangents.append([tx, ty, tz])

            # Bitangent = ∂P/∂v (normalised) = ∂P/∂θ · π
            # ∂P/∂θ = (cos_t·cos_p, -sin_t, cos_t·sin_p)
            bx = cos_t * cos_p
            by = -sin_t
            bz = cos_t * sin_p
            lb = math.sqrt(bx*bx + by*by + bz*bz)
            if lb > 1e-6:
                bx, by, bz = bx/lb, by/lb, bz/lb
            bitangents.append([bx, by, bz])

    # Build face index list (two triangles per quad)
    faces = []
    for ri in range(rings):
        for si in range(sectors):
            row_len = sectors + 1
            tl = ri * row_len + si
            tr = tl + 1
            bl = tl + row_len
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    return Mesh(
        verts=np.array(verts,     dtype=np.float32),
        uvs=np.array(uvs,         dtype=np.float32),
        normals=np.array(normals,  dtype=np.float32),
        tangents=np.array(tangents, dtype=np.float32),
        bitangents=np.array(bitangents, dtype=np.float32),
        faces=np.array(faces,     dtype=np.int32),
    )
