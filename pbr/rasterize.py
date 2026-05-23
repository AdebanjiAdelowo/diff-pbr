"""Software rasterizer producing per-pixel geometry buffers.

For each camera view this module outputs fixed (non-differentiable) buffers:
    pixel_uv      (H, W, 2)  UV coordinates at each pixel
    pixel_normal  (H, W, 3)  interpolated world-space normal
    pixel_tangent (H, W, 3)  interpolated world-space tangent
    pixel_bitangent(H,W, 3)  interpolated world-space bitangent
    pixel_pos     (H, W, 3)  interpolated world-space position
    pixel_mask    (H, W)     bool — True where the surface was hit

Rasterisation uses per-triangle barycentric interpolation and a
z-buffer.  Runs on CPU with numpy only; called once per view before
the differentiable pass.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .mesh import Mesh
from .camera import Camera


@dataclass
class GeometryBuffers:
    pixel_uv:        np.ndarray   # (H, W, 2)  float32
    pixel_normal:    np.ndarray   # (H, W, 3)  float32
    pixel_tangent:   np.ndarray   # (H, W, 3)  float32
    pixel_bitangent: np.ndarray   # (H, W, 3)  float32
    pixel_pos:       np.ndarray   # (H, W, 3)  float32
    pixel_mask:      np.ndarray   # (H, W)     bool


def rasterize(mesh: Mesh, cam: Camera) -> GeometryBuffers:
    """CPU rasterizer.  Returns geometry buffers for one camera view."""
    H, W = cam.height, cam.width
    V    = cam.view_matrix()       # (4,4)
    P    = cam.projection_matrix() # (4,4)
    VP   = P @ V                   # (4,4)

    # Transform all vertices to clip space
    v4 = np.concatenate([mesh.verts, np.ones((len(mesh.verts), 1), np.float32)], axis=1)
    clip = (VP @ v4.T).T           # (V, 4)
    w    = clip[:, 3:4]
    ndc  = clip[:, :3] / (np.abs(w) + 1e-8)   # (V, 3) NDC x,y,z

    # NDC → screen space (integer pixel coords)
    sx = ((ndc[:, 0] + 1.0) * 0.5 * (W - 1))   # (V,)
    sy = ((1.0 - ndc[:, 1]) * 0.5 * (H - 1))   # flip y for image coords

    # Output buffers
    z_buf   = np.full((H, W), np.inf, dtype=np.float32)
    uv_buf  = np.zeros((H, W, 2),  dtype=np.float32)
    n_buf   = np.zeros((H, W, 3),  dtype=np.float32)
    tan_buf = np.zeros((H, W, 3),  dtype=np.float32)
    bit_buf = np.zeros((H, W, 3),  dtype=np.float32)
    pos_buf = np.zeros((H, W, 3),  dtype=np.float32)

    for face in mesh.faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

        # Back-face cull in screen space (y-down convention).
        # Positive cross = CCW in world (back-facing for outward-normals sphere).
        e1x = sx[i1] - sx[i0]; e1y = sy[i1] - sy[i0]
        e2x = sx[i2] - sx[i0]; e2y = sy[i2] - sy[i0]
        if (e1x * e2y - e1y * e2x) <= 0:
            continue

        # Bounding box (clamped to screen)
        xs = np.array([sx[i0], sx[i1], sx[i2]])
        ys = np.array([sy[i0], sy[i1], sy[i2]])
        x0 = max(0,   int(np.floor(xs.min())))
        x1 = min(W-1, int(np.ceil( xs.max())))
        y0 = max(0,   int(np.floor(ys.min())))
        y1 = min(H-1, int(np.ceil( ys.max())))

        if x0 > x1 or y0 > y1:
            continue

        # Rasterise all pixels in bounding box via barycentric test
        gx, gy = np.meshgrid(np.arange(x0, x1+1, dtype=np.float32),
                              np.arange(y0, y1+1, dtype=np.float32))
        # Signed areas
        def edge(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

        w0 = edge(sx[i1], sy[i1], sx[i2], sy[i2], gx, gy)
        w1 = edge(sx[i2], sy[i2], sx[i0], sy[i0], gx, gy)
        w2 = edge(sx[i0], sy[i0], sx[i1], sy[i1], gx, gy)

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        denom = w0 + w1 + w2
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        b0 = w0 / denom
        b1 = w1 / denom
        b2 = w2 / denom

        # Perspective-correct depth interpolation (NDC z)
        z = b0 * ndc[i0, 2] + b1 * ndc[i1, 2] + b2 * ndc[i2, 2]

        py_idx = np.arange(y0, y1+1)[:, None].repeat(x1-x0+1, axis=1)
        px_idx = np.arange(x0, x1+1)[None, :].repeat(y1-y0+1, axis=0)

        update = inside & (z < z_buf[py_idx, px_idx])
        if not update.any():
            continue

        py_sel = py_idx[update]
        px_sel = px_idx[update]
        b0s = b0[update, np.newaxis]
        b1s = b1[update, np.newaxis]
        b2s = b2[update, np.newaxis]

        z_buf[py_sel, px_sel]    = z[update]
        uv_buf[py_sel,  px_sel]  = (b0s * mesh.uvs[i0]
                                  + b1s * mesh.uvs[i1]
                                  + b2s * mesh.uvs[i2])
        n_buf[py_sel,   px_sel]  = (b0s * mesh.normals[i0]
                                  + b1s * mesh.normals[i1]
                                  + b2s * mesh.normals[i2])
        tan_buf[py_sel, px_sel]  = (b0s * mesh.tangents[i0]
                                  + b1s * mesh.tangents[i1]
                                  + b2s * mesh.tangents[i2])
        bit_buf[py_sel, px_sel]  = (b0s * mesh.bitangents[i0]
                                  + b1s * mesh.bitangents[i1]
                                  + b2s * mesh.bitangents[i2])
        pos_buf[py_sel, px_sel]  = (b0s * mesh.verts[i0]
                                  + b1s * mesh.verts[i1]
                                  + b2s * mesh.verts[i2])

    mask = np.isfinite(z_buf)
    # Re-normalise interpolated vectors
    for buf in (n_buf, tan_buf, bit_buf):
        norms = np.linalg.norm(buf, axis=-1, keepdims=True).clip(min=1e-8)
        buf /= norms

    return GeometryBuffers(
        pixel_uv=uv_buf,
        pixel_normal=n_buf,
        pixel_tangent=tan_buf,
        pixel_bitangent=bit_buf,
        pixel_pos=pos_buf,
        pixel_mask=mask,
    )
