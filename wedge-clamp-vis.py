"""Wedge-clamp axial-section profile rendered as a donut.c-style point cloud.

Geometry (axisymmetric, mirrored about z = 0):

    upper truncated cone   r at z = +(offset + h),  R at z = +offset
    cylinder, radius R     z in [-offset, +offset]   (height = 2 * offset)
    lower truncated cone   R at z = -offset,         r at z = -(offset + h)

with h = (R - r) / tan(theta).

The 2D right-half cross-section is sampled on a regular grid; points
inside the profile are projected to pixels and shaded by their radial
distance from the axis. Boundary points (cylinder walls, cone slants,
cap rims) get full brightness; brightness fades to ambient on the axis.
The image canvas is sized so the geometry's aspect ratio is preserved.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# USER PARAMETERS — edit these and re-run; CLI flags still override.
# =============================================================================
# Geometry
r          = 40.0    # inner (small) cone radius
R          = 50.0    # outer (large) cone radius
theta_deg  = 30.0    # cone half-angle, degrees (ignored if h is set)
h          = None    # axial height of one cone; if set, overrides theta_deg
offset     = 5.0     # half-height of the central cylinder (height = 2*offset)
clearance  = 1.0     # gap between wedge surfaces and the approaching-shape mating face
shape_w    = 50.0    # rectangle width of each approaching shape

# Animation (frames=1 -> static PNG; frames>1 -> GIF of shapes closing on the wedge)
frames     = 60      # number of animation frames
fps        = 30      # GIF frame rate
travel     = 50.0    # how far each shape starts outside its mating position

# Layout
header_px  = 80      # extra pixels reserved at the top for the HUD text band
# =============================================================================


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


def max_radius_at_z(z: np.ndarray, r: float, R: float, h_cone: float, offset: float) -> np.ndarray:
    """Return the local boundary radius for each z, or -1 outside the profile."""
    abs_z = np.abs(z)
    in_cyl = abs_z <= offset
    in_cone = (abs_z > offset) & (abs_z <= offset + h_cone)
    d = np.where(in_cone, abs_z - offset, 0.0)
    cone_rad = R - (R - r) * d / h_cone
    return np.where(in_cyl, R, np.where(in_cone, cone_rad, -1.0))


def notch_face_x(z: np.ndarray, x_inner: float, x_outer: float,
                 h_cone: float, offset: float) -> np.ndarray:
    """Mating-face x of an approaching shape at each z (notch profile).

    The profile follows the wedge's silhouette offset by `clearance`, encoded
    by the caller into x_inner / x_outer:
        |z| <= offset            -> deep flat at x_inner
        offset < |z| <= offset+h -> angled, linear from x_inner to x_outer

    Works for both shapes: pass (x_inner_L, x_outer_L) for the left shape (the
    function returns its right boundary) or (x_inner_R, x_outer_R) for the
    right shape (function returns its left boundary). Inner/outer values can
    be negative or positive — only their (signed) ordering matters.
    """
    abs_z = np.abs(z)
    angled_frac = np.clip((abs_z - offset) / h_cone, 0.0, 1.0)
    return np.where(abs_z <= offset, x_inner, x_inner + (x_outer - x_inner) * angled_frac)


def ease_in_out(t: float) -> float:
    """Smoothstep easing — slows the shapes near both endpoints."""
    return t * t * (3.0 - 2.0 * t)


def compute_canvas(args: argparse.Namespace, z_top: float):
    """Pick canvas dimensions sized for the widest frame (shapes at t=0).

    Returns (width, height, scale, cy_geom). A fixed-height ``header_px``
    band is added on top of the geometry so the HUD text sits clear of
    the shapes; the geometry's vertical centre (z=0) lands at cy_geom.
    """
    extra = max(args.travel, 0.0) if args.frames > 1 else 0.0
    x_max = args.r + args.clearance + extra + args.shape_w
    world_w = 2.0 * x_max
    world_h = 2.0 * z_top

    canvas_w_world = world_w / max(1e-9, 1.0 - 2.0 * args.pad_x)
    canvas_h_world = world_h / max(1e-9, 1.0 - 2.0 * args.pad_y)
    if canvas_w_world >= canvas_h_world:
        scale = args.long_px / canvas_w_world
        width = args.long_px
        geom_height = int(round(canvas_h_world * scale))
    else:
        scale = args.long_px / canvas_h_world
        geom_height = args.long_px
        width = int(round(canvas_w_world * scale))

    header = max(0, args.header_px)
    height = geom_height + header
    cy_geom = header + geom_height / 2.0
    return width, height, scale, cy_geom


def _shape_shade(x_world, z_world, x_inner, x_outer, x_far, h_cone, offset, z_top, args):
    """Shade one approaching shape.

    x_far is the rectangle's far edge (left edge for left shape, right edge for
    right shape). x_inner/x_outer follow the convention of notch_face_x.
    """
    face = notch_face_x(z_world, x_inner, x_outer, h_cone, offset)
    if x_far < x_outer:
        # Left shape: interior is x in [x_far, face].
        x_lo, x_hi = x_far, face
    else:
        # Right shape: interior is x in [face, x_far].
        x_lo, x_hi = face, x_far

    inside = ((z_world >= -z_top) & (z_world <= z_top)
              & (x_world >= x_lo) & (x_world <= x_hi))
    center = 0.5 * (x_lo + x_hi)
    half_w = np.maximum(0.5 * (x_hi - x_lo), 1e-9)
    horiz_t = np.abs(x_world - center) / half_w
    vert_t  = np.abs(z_world) / max(z_top, 1e-9)
    bright = np.clip(np.maximum(horiz_t, vert_t), 0.0, 1.0) ** args.brightness_pow
    shade = (args.ambient + (1.0 - args.ambient) * bright) ** args.gamma
    return np.where(inside, shade, 0.0)


def render_frame(
    args: argparse.Namespace, t: float, *,
    width: int, height: int, scale: float, x_center_world: float,
    cy_geom: float,
    PX: np.ndarray, PY: np.ndarray, geom: dict[str, float],
) -> np.ndarray:
    """Render one frame at animation phase t in [0,1].

    t=0: shapes at the far end of `travel` outside their mating positions.
    t=1: shapes at the mating positions (clearance from R).
    """
    r, R, offset = args.r, args.R, args.offset
    h_cone = geom["h"]
    z_top = offset + h_cone

    extra = args.travel * (1.0 - ease_in_out(float(np.clip(t, 0.0, 1.0))))
    # Both shapes drift inward symmetrically; mating geometry is `extra=0`.
    x_inner_L = -(R + args.clearance + extra)
    x_outer_L = -(r + args.clearance + extra)
    x_far_L   = x_outer_L - args.shape_w

    x_inner_R = +(R + args.clearance + extra)
    x_outer_R = +(r + args.clearance + extra)
    x_far_R   = x_outer_R + args.shape_w

    cx = width / 2.0
    x_world = (PX - cx) / scale + x_center_world
    z_world = (cy_geom - PY) / scale

    # ---- wedge: bright at radial boundary, dim on axis ----
    rad_world = np.abs(x_world)
    max_rad = max_radius_at_z(z_world, r, R, h_cone, offset)
    in_wedge = (max_rad > 0.0) & (rad_world <= max_rad)
    safe_max = np.where(max_rad > 1e-12, max_rad, 1.0)
    wedge_bright = np.clip(rad_world / safe_max, 0.0, 1.0) ** args.brightness_pow
    wedge_shade = (args.ambient + (1.0 - args.ambient) * wedge_bright) ** args.gamma
    wedge_shade = np.where(in_wedge, wedge_shade, 0.0)

    # ---- two approaching shapes ----
    shade_L = _shape_shade(x_world, z_world, x_inner_L, x_outer_L, x_far_L,
                           h_cone, offset, z_top, args)
    shade_R = _shape_shade(x_world, z_world, x_inner_R, x_outer_R, x_far_R,
                           h_cone, offset, z_top, args)

    if args.dot_period > 1:
        dot_mask = ((PX % args.dot_period) == 0) & ((PY % args.dot_period) == 0)
        wedge_shade = np.where(dot_mask, wedge_shade, 0.0)
        shade_L = np.where(dot_mask, shade_L, 0.0)
        shade_R = np.where(dot_mask, shade_R, 0.0)

    wedge_color = np.array([1.00, 0.55, 0.25], dtype=np.float64)  # warm orange
    shape_color = np.array([0.45, 0.65, 0.95], dtype=np.float64)  # cool blue
    img = (wedge_shade[..., None] * wedge_color
           + shade_L[..., None] * shape_color
           + shade_R[..., None] * shape_color)
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if not args.no_hud:
        arr = _overlay_hud(arr, args, geom)
    return arr


def _overlay_hud(arr: np.ndarray, args: argparse.Namespace, geom: dict[str, float]) -> np.ndarray:
    p_mean = mean_pressure(args.P, args.theta, args.mu, args.r, args.R)
    N_force = normal_force(args.P, args.theta, args.mu)
    lines = [
        f"r={args.r:g}  R={args.R:g}  theta={math.degrees(args.theta):.2f} deg  offset={args.offset:g}",
        f"h={geom['h']:.4g}  l={geom['l']:.4g}",
        f"A_L={geom['A_L']:.4g}  A_t={geom['A_t']:.4g}  A_b={geom['A_b']:.4g}",
        f"P={args.P:g}  mu={args.mu:g}  N={N_force:.4g}  p_mean={p_mean:.4g}",
    ]
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r", type=float, default=r, help="inner (small) cone radius")
    p.add_argument("--R", type=float, default=R, help="outer (large) cone radius")
    p.add_argument("--theta", type=float, default=None,
                   help="cone half-angle in radians (overrides --theta-deg)")
    p.add_argument("--theta-deg", type=float, default=theta_deg, help="cone half-angle in degrees")
    p.add_argument("--h", type=float, default=h,
                   help="axial height of one cone; if set, overrides theta")
    p.add_argument("--offset", type=float, default=offset,
                   help="half-height of central cylinder; cylinder height = 2 * offset")
    p.add_argument("--clearance", type=float, default=clearance,
                   help="gap between wedge surfaces and approaching-shape mating face")
    p.add_argument("--shape-w", type=float, default=shape_w,
                   help="rectangle width of each approaching shape")
    p.add_argument("--frames", type=int, default=frames,
                   help="number of animation frames (1 = static PNG, >1 = GIF)")
    p.add_argument("--fps", type=int, default=fps, help="GIF frame rate")
    p.add_argument("--travel", type=float, default=travel,
                   help="how far each shape starts outside its mating position")
    p.add_argument("--header-px", type=int, default=header_px,
                   help="pixels reserved at top of canvas for HUD text (0 = no header)")
    p.add_argument("--P", type=float, default=1000.0, help="applied axial load")
    p.add_argument("--mu", type=float, default=0.15, help="friction coefficient")
    p.add_argument("--long-px", type=int, default=900,
                   help="pixels along the longest world dimension")
    p.add_argument("--pad-x", type=float, default=0.25,
                   help="fraction of canvas reserved as margin on EACH horizontal side "
                        "(default 0.25 = 50%% total side padding)")
    p.add_argument("--pad-y", type=float, default=0.15,
                   help="fraction of canvas reserved as margin on EACH vertical side "
                        "(default 0.15 = 30%% total vertical padding)")
    p.add_argument("--dot-period", type=int, default=1,
                   help="if >1, only every Nth pixel is lit, giving an explicit dot grid")
    p.add_argument("--brightness-pow", type=float, default=1.5,
                   help="exponent applied to (radius / boundary_radius); >1 darkens interior")
    p.add_argument("--ambient", type=float, default=0.10,
                   help="floor brightness on the axis")
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--out", default="wedge-clamp-profile.png")
    args = p.parse_args()
    if args.r >= args.R:
        p.error("require r < R")
    if args.h is not None:
        if args.h <= 0.0:
            p.error("h must be > 0")
        args.theta = math.atan2(args.R - args.r, args.h)
    elif args.theta is None:
        args.theta = math.radians(args.theta_deg)
    if not (0.0 < args.theta < math.pi / 2):
        p.error("theta must be in (0, pi/2)")
    if args.offset < 0.0:
        p.error("offset must be >= 0")
    if args.clearance < 0.0:
        p.error("clearance must be >= 0")
    if args.shape_w <= 0.0:
        p.error("shape_w must be > 0")
    return args


def main() -> None:
    args = parse_args()
    geom = cone_geometry(args.r, args.R, args.theta)
    z_top = args.offset + geom["h"]

    width, height, scale, cy_geom = compute_canvas(args, z_top)
    PX, PY = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    # The assembly is symmetric around the wedge axis (x = 0).
    x_center_world = 0.0

    out_path = Path(args.out).resolve()

    if args.frames <= 1:
        arr = render_frame(args, t=1.0, width=width, height=height, scale=scale,
                           x_center_world=x_center_world, cy_geom=cy_geom,
                           PX=PX, PY=PY, geom=geom)
        if out_path.suffix.lower() == ".gif":
            out_path = out_path.with_suffix(".png")
        Image.fromarray(arr).save(out_path)
        print(f"wrote {out_path}  ({width}x{height} px, h={geom['h']:.4g}, l={geom['l']:.4g})")
        return

    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    pil_frames: list[Image.Image] = []
    for i in range(args.frames):
        t = i / max(1, args.frames - 1)
        arr = render_frame(args, t, width=width, height=height, scale=scale,
                           x_center_world=x_center_world, cy_geom=cy_geom,
                           PX=PX, PY=PY, geom=geom)
        pil_frames.append(Image.fromarray(arr))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(round(1000.0 / args.fps)),
        loop=0,
        disposal=2,
    )
    print(f"wrote {out_path}  ({width}x{height} px, {args.frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
