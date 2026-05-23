"""Camera and light utilities for the diff-pbr renderer.

All coordinate frames use right-handed Y-up conventions.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Camera:
    """Pinhole camera."""
    position:    np.ndarray   # (3,)  world-space eye position
    target:      np.ndarray   # (3,)  look-at point
    up:          np.ndarray   # (3,)  up hint
    fov_y:       float        # vertical field-of-view in degrees
    width:       int
    height:      int

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def view_matrix(self) -> np.ndarray:
        """4×4 world→view matrix (right-handed, column-major convention)."""
        z = self.position - self.target
        z /= np.linalg.norm(z) + 1e-10
        x = np.cross(self.up, z)
        x /= np.linalg.norm(x) + 1e-10
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)          # (3,3)
        t = -R @ self.position                    # (3,)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3]  = t
        return M

    def projection_matrix(self) -> np.ndarray:
        """4×4 perspective projection (NDC: x,y ∈ [-1,1], z ∈ [-1,1])."""
        n, f = 0.01, 100.0
        t_half = math.tan(math.radians(self.fov_y / 2.0))
        sx = 1.0 / (self.aspect * t_half)
        sy = 1.0 / t_half
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = sx
        P[1, 1] = sy
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = -1.0
        return P


@dataclass
class PointLight:
    position: np.ndarray   # (3,) world space
    color:    np.ndarray   # (3,) linear RGB intensity


def hemisphere_cameras(
    n: int = 8,
    radius: float = 3.5,
    elevation_deg: float = 30.0,
    img_w: int = 256,
    img_h: int = 256,
    fov_y: float = 35.0,
) -> List[Camera]:
    """Evenly spaced cameras on a hemisphere at a given elevation."""
    cameras = []
    for i in range(n):
        phi = 2.0 * math.pi * i / n
        theta = math.radians(90.0 - elevation_deg)
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.cos(theta)
        z = radius * math.sin(theta) * math.sin(phi)
        cameras.append(Camera(
            position=np.array([x, y, z], dtype=np.float32),
            target=np.zeros(3, dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            fov_y=fov_y,
            width=img_w,
            height=img_h,
        ))
    return cameras


def default_lights() -> List[PointLight]:
    """Three-point lighting rig."""
    return [
        PointLight(np.array([ 4.0,  4.0,  4.0], np.float32),
                   np.array([1.2,  1.1,  1.0],  np.float32)),
        PointLight(np.array([-4.0,  2.0,  3.0], np.float32),
                   np.array([0.5,  0.55, 0.6],  np.float32)),
        PointLight(np.array([ 0.0, -3.0, -4.0], np.float32),
                   np.array([0.25, 0.25, 0.3],  np.float32)),
    ]
