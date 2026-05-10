"""3D truncated-cone visualizer in the style of donut.c (a1k0n.net/2011/07/20/donut-math).

Surface is rendered as a point cloud with Lambertian shading, animated by
spinning the cone around its axis (z) and writing an animated GIF.

Geometry parametrised by inner radius r, outer radius R, half-angle theta:

    h   = (R - r) / tan(theta)                              axial height
    l   = (R - r) / sin(theta)                              slant length
    A_L = pi (r + R) l                                      lateral contact area
    A_t = pi r^2,   A_b = pi R^2                            cap areas
    N   = P / [2 (sin theta + mu cos theta)]                interface normal force
    p   = P sin(theta)
          / [2 pi (r + R)(R - r)(sin theta + mu cos theta)] mean lateral pressure
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cone_geometry(r: float, R: float, theta: float) -> dict[str, float]:
    h = (R - r) / math.tan(theta)
    l = (R - r) / math.sin(theta)
    return {
        "h": h,
        "l": l,
        "A_L": math.pi * (r + R) * l,
        "A_t": math.pi * r * r,
        "A_b": math.pi * R * R,
    }


def mean_pressure(P: float, theta: float, mu: float, r: float, R: float) -> float:
    return (
        P * math.sin(theta)
        / (2 * math.pi * (r + R) * (R - r) * (math.sin(theta) + mu * math.cos(theta)))
    )


def normal_force(P: float, theta: float, mu: float) -> float:
    return P / (2 * (math.sin(theta) + mu * math.cos(theta)))


def sample_cone(
    r: float, R: float, theta: float, *,
    n_slant: int, n_circ: int, n_cap: int,
):
    """Return [(name, points, normals)] for lateral, top cap, bottom cap.

    Cone axis is +z; bottom cap (radius R) at z=0, top cap (radius r) at z=h.
    """
    h = (R - r) / math.tan(theta)
    phi = np.linspace(0.0, 2 * np.pi, n_circ, endpoint=False)

    s = np.linspace(0.0, 1.0, n_slant)
    S, Ph = np.meshgrid(s, phi, indexing="ij")
    rad = R - (R - r) * S
    z = h * S
    lateral_pts = np.column_stack([
        (rad * np.cos(Ph)).ravel(),
        (rad * np.sin(Ph)).ravel(),
        z.ravel(),
    ])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    lateral_nrm = np.column_stack([
        (np.cos(Ph) * cos_t).ravel(),
        (np.sin(Ph) * cos_t).ravel(),
        np.full(Ph.size, sin_t),
    ])

    rho_t = np.linspace(0.0, r, n_cap)
    Rt, Pt = np.meshgrid(rho_t, phi, indexing="ij")
    top_pts = np.column_stack([
        (Rt * np.cos(Pt)).ravel(),
        (Rt * np.sin(Pt)).ravel(),
        np.full(Rt.size, h),
    ])
    top_nrm = np.tile(np.array([0.0, 0.0, 1.0]), (top_pts.shape[0], 1))

    rho_b = np.linspace(0.0, R, n_cap)
    Rb, Pb = np.meshgrid(rho_b, phi, indexing="ij")
    bot_pts = np.column_stack([
        (Rb * np.cos(Pb)).ravel(),
        (Rb * np.sin(Pb)).ravel(),
        np.zeros(Rb.size),
    ])
    bot_nrm = np.tile(np.array([0.0, 0.0, -1.0]), (bot_pts.shape[0], 1))

    return [
        ("lateral", lateral_pts, lateral_nrm),
        ("top", top_pts, top_nrm),
        ("bottom", bot_pts, bot_nrm),
    ]


def render_frame(
    parts, alpha, *,
    width: int, height: int, lean: float,
    K1: float, K2: float, light_dir, palette, center_z: float,
) -> np.ndarray:
    """Render one frame.

    World convention: x right, y depth (away from camera at -y), z up.
    Cone-local axis is z. The cone is leaned by ``lean`` around world-x
    (top tilts toward camera) and then spun by ``alpha`` around world-z;
    a perfect cone is symmetric about its axis, so without the lean the
    z-spin would produce no visible motion.
    """
    cl, sl = math.cos(lean), math.sin(lean)
    Rx_lean = np.array([[1.0, 0.0, 0.0],
                        [0.0,  cl, -sl],
                        [0.0,  sl,  cl]])
    ca, sa = math.cos(alpha), math.sin(alpha)
    Rz_spin = np.array([[ca, -sa, 0.0],
                        [sa,  ca, 0.0],
                        [0.0, 0.0, 1.0]])
    # World -> camera: world-x stays right, world-z becomes camera-up,
    # world-y becomes camera-depth.
    M_perm = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]])
    M_world = Rz_spin @ Rx_lean
    M_camera = M_perm @ M_world

    light_world = np.asarray(light_dir, dtype=np.float64)
    light_world /= np.linalg.norm(light_world)

    pts_list, nrm_list, col_list = [], [], []
    for name, pts, nrm in parts:
        p = pts.copy()
        p[:, 2] -= center_z
        pts_list.append(p)
        nrm_list.append(nrm)
        col_list.append(np.tile(palette[name], (pts.shape[0], 1)))
    P = np.concatenate(pts_list)
    N = np.concatenate(nrm_list)
    C = np.concatenate(col_list)

    P_cam = P @ M_camera.T
    N_world = N @ M_world.T  # lighting computed in world frame
    z_eff = P_cam[:, 2] + K2

    safe_z = np.where(z_eff > 1e-3, z_eff, 1.0)
    ooz = 1.0 / safe_z
    cx, cy = width / 2.0, height / 2.0
    sx = (cx + K1 * P_cam[:, 0] * ooz).astype(np.int64)
    sy = (cy - K1 * P_cam[:, 1] * ooz).astype(np.int64)
    diffuse = N_world @ light_world

    # Backface cull: camera at -y world, so a surface is camera-facing iff
    # its world normal has y < 0 (orthographic approximation; safe for K2
    # >> object size).
    camera_facing = N_world[:, 1] < 0.0

    visible = (
        (z_eff > 1e-3)
        & camera_facing
        & (sx >= 0) & (sx < width)
        & (sy >= 0) & (sy < height)
    )
    sx = sx[visible]; sy = sy[visible]
    diffuse = diffuse[visible]
    ooz = ooz[visible]
    col = C[visible]

    # Painter's algorithm via fancy-index assign: sort ascending by ooz so
    # closer points (larger ooz) are written last and overwrite farther ones.
    order = np.argsort(ooz, kind="stable")
    sx = sx[order]; sy = sy[order]
    diffuse = diffuse[order]; col = col[order]

    # Ambient + clamped Lambertian, mild gamma for richer mids.
    shade = 0.12 + 0.88 * np.clip(diffuse, 0.0, 1.0)
    shade = shade ** 0.85

    img = np.zeros((height, width, 3), dtype=np.float64)
    img[sy, sx] = shade[:, None] * col
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def overlay_hud(arr: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.text((9, y + 1), line, fill=(0, 0, 0), font=font)
        draw.text((8, y),     line, fill=(225, 225, 230), font=font)
        y += 15
    return np.asarray(img)


def build_gif(args: argparse.Namespace) -> Path:
    parts = sample_cone(
        args.r, args.R, args.theta,
        n_slant=args.n_slant, n_circ=args.n_circ, n_cap=args.n_cap,
    )
    geom = cone_geometry(args.r, args.R, args.theta)
    p_mean = mean_pressure(args.P, args.theta, args.mu, args.r, args.R)
    N_force = normal_force(args.P, args.theta, args.mu)
    center_z = geom["h"] / 2.0

    lean_rad = math.radians(args.lean)
    h_extent = 2.0 * args.R
    v_extent = geom["h"] * math.cos(lean_rad) + (args.R + args.r) * math.sin(lean_rad)
    extent = max(h_extent, v_extent)
    K2 = 4.0 * extent
    K1 = 0.8 * min(args.width, args.height) * K2 / extent

    palette = {
        "lateral": np.array([1.00, 0.55, 0.25]),
        "top":     np.array([0.85, 0.85, 0.95]),
        "bottom":  np.array([0.45, 0.65, 0.95]),
    }

    hud = [
        f"r={args.r:.4g}  R={args.R:.4g}  theta={math.degrees(args.theta):.2f} deg  mu={args.mu:.3g}",
        f"h={geom['h']:.4g}  l={geom['l']:.4g}",
        f"A_L={geom['A_L']:.4g}  A_t={geom['A_t']:.4g}  A_b={geom['A_b']:.4g}",
        f"P={args.P:.4g}  N={N_force:.4g}  p_mean={p_mean:.4g}",
    ]

    frames: list[Image.Image] = []
    for i in range(args.frames):
        alpha = 2.0 * math.pi * i / args.frames
        arr = render_frame(
            parts, alpha,
            width=args.width, height=args.height,
            lean=lean_rad,
            K1=K1, K2=K2,
            light_dir=(0.55, -0.55, 0.6),
            palette=palette,
            center_z=center_z,
        )
        if not args.no_hud:
            arr = overlay_hud(arr, hud)
        frames.append(Image.fromarray(arr))

    out_path = Path(args.out).resolve()
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / args.fps)),
        loop=0,
        disposal=2,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r", type=float, default=1.0, help="inner (small) radius")
    p.add_argument("--R", type=float, default=2.0, help="outer (large) radius")
    p.add_argument("--theta", type=float, default=None,
                   help="half-angle in radians (overrides --theta-deg)")
    p.add_argument("--theta-deg", type=float, default=30.0, help="half-angle in degrees")
    p.add_argument("--P", type=float, default=1000.0, help="applied axial load")
    p.add_argument("--mu", type=float, default=0.15, help="friction coefficient")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--lean", type=float, default=22.0,
                   help="cone lean off world-z, in degrees (>0 makes z-spin visible)")
    p.add_argument("--n-slant", type=int, default=140)
    p.add_argument("--n-circ", type=int, default=480)
    p.add_argument("--n-cap", type=int, default=32)
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--out", default="cone.gif")
    args = p.parse_args()
    if args.theta is None:
        args.theta = math.radians(args.theta_deg)
    if not (0.0 < args.theta < math.pi / 2):
        p.error("theta must be in (0, pi/2)")
    if args.r >= args.R:
        p.error("require r < R")
    return args


def main() -> None:
    out = build_gif(parse_args())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
