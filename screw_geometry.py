"""Deterministic CPU screw silhouette alignment, using image pixels only.

No labels, defect masks, pretrained segmentation model or new dependencies are
used. Geometry is estimated at 256 px, but the RGB image is resampled once at
its original size. Quality checks measure geometric stability, not correctness
of anomaly detection. A failed check uses an explicitly recorded identity
transform. Experiments must report that fallback coverage separately.
"""
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


GEOMETRY_VERSION = "otsu-lcc-pca-head-width-256-axis-bbox-v2"
QUALITY_RULE = {
    "min_area_fraction": .02,
    "max_area_fraction": .65,
    "min_component_fraction": .60,
    "min_anisotropy": 3.,
    "min_end_width_ratio": 1.3,
    "max_threshold_angle_change_degrees": 5.,
    "max_threshold_center_shift_px256": 5.,
    "max_border_contact": 0,
    "min_foreground_retained_fraction": .995,
}


def _gray(image):
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if min(image.size) < 2:
        raise ValueError("image must be at least 2 by 2 pixels")
    return np.asarray(image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
                      .filter(ImageFilter.GaussianBlur(1)))


def _otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    weight = hist.cumsum()
    total = weight[-1]
    cumulative = (hist * np.arange(256)).cumsum()
    denominator = weight * (total - weight)
    score = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    score[valid] = (cumulative[-1] * weight[valid] - total * cumulative[valid]) ** 2 / denominator[valid]
    return int(score.argmax())


def _largest_component(mask):
    """Largest 4-connected component; deterministic top-left tie breaking."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    winner = []
    for flat in np.flatnonzero(mask):
        y, x = divmod(int(flat), width)
        if seen[y, x]:
            continue
        queue = deque([(y, x)])
        seen[y, x] = True
        points = []
        while queue:
            yy, xx = queue.popleft()
            points.append((yy, xx))
            for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(points) > len(winner):
            winner = points
    result = np.zeros_like(mask)
    if winner:
        coords = np.asarray(winner)
        result[coords[:, 0], coords[:, 1]] = True
    return result


def _describe(gray, threshold):
    raw = gray <= np.clip(threshold, 0, 255)
    mask = _largest_component(raw)
    y, x = np.nonzero(mask)
    diagnostics = {
        "threshold": int(threshold),
        "area_fraction": float(mask.mean()),
        "largest_component_fraction": float(mask.sum() / max(1, raw.sum())),
        "border_contact": int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()),
        "geometry_available": False,
    }
    if len(x) < 16:
        return mask, diagnostics
    xy = np.stack((x, y), axis=1).astype(np.float64)
    center = xy.mean(axis=0)
    centered = xy - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(xy))
    axis = eigenvectors[:, -1]
    side = np.array([-axis[1], axis[0]])
    longitudinal = centered @ axis
    transverse = centered @ side
    lo, hi = np.quantile(longitudinal, [.01, .99])
    span = float(hi - lo)
    low = transverse[(longitudinal >= lo) & (longitudinal < lo + .20 * span)]
    high = transverse[(longitudinal <= hi) & (longitudinal > hi - .20 * span)]
    if len(low) < 2 or len(high) < 2:
        return mask, diagnostics
    low_width = float(np.quantile(low, .95) - np.quantile(low, .05))
    high_width = float(np.quantile(high, .95) - np.quantile(high, .05))
    head_axis = axis if high_width > low_width else -axis
    diagnostics.update({
        "geometry_available": True,
        "center_px256": center.tolist(),
        "alignment_center_px256": (center + axis * .5 * (longitudinal.min() + longitudinal.max())).tolist(),
        "head_axis": head_axis.tolist(),
        "head_angle_degrees": float(np.degrees(np.arctan2(head_axis[1], head_axis[0]))),
        "anisotropy": float(eigenvalues[-1] / max(eigenvalues[0], 1e-9)),
        "axis_span_px256": span,
        "end_width_ratio": max(low_width, high_width) / max(1e-9, min(low_width, high_width)),
        "head_width_px256": max(low_width, high_width),
        "tip_width_px256": min(low_width, high_width),
    })
    return mask, diagnostics


def foreground_mask(image):
    """Return original-size bool silhouette and JSON-compatible diagnostics.

    The silhouette is a nearest-neighbor enlargement of a 256 px estimate. It
    is intended for geometric preprocessing/synthetic experiment placement,
    not as a precise segmentation or defect ground truth.
    """
    gray = _gray(image)
    small, diagnostics = _describe(gray, _otsu(gray))
    enlarged = Image.fromarray(small.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    diagnostics["version"] = GEOMETRY_VERSION
    diagnostics["source_size"] = list(image.size)
    return np.asarray(enlarged) > 0, diagnostics


def _angle_difference(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def _quality_failures(info):
    rule = QUALITY_RULE
    reasons = []
    if not info["geometry_available"]:
        return ["insufficient_foreground_geometry"]
    checks = (
        (info["area_fraction"] < rule["min_area_fraction"], "foreground_too_small"),
        (info["area_fraction"] > rule["max_area_fraction"], "foreground_too_large"),
        (info["largest_component_fraction"] < rule["min_component_fraction"], "foreground_fragmented"),
        (info["anisotropy"] < rule["min_anisotropy"], "axis_not_distinct"),
        (info["end_width_ratio"] < rule["min_end_width_ratio"], "head_tip_ambiguous"),
        (info["border_contact"] > rule["max_border_contact"], "foreground_touches_border"),
        (info["threshold_perturbation_max_angle_degrees"] > rule["max_threshold_angle_change_degrees"], "direction_unstable"),
        (info["threshold_perturbation_max_center_shift_px256"] > rule["max_threshold_center_shift_px256"], "center_unstable"),
        (info["foreground_retained_fraction"] < rule["min_foreground_retained_fraction"], "alignment_clips_foreground"),
    )
    for failed, reason in checks:
        if failed:
            reasons.append(reason)
    return reasons


def alignment_transform(image):
    """Return aligned original-size RGB PIL image and reversible metadata.

    Wide end is placed upwards; the midpoint of the full axial foreground
    extent is centered along the screw axis, with the transverse centroid
    unchanged. This avoids clipping tips due to the heavier, wider screw head.
    There is no crop or scaling. RGB uses one bilinear affine resampling.
    ``forward_affine`` maps original to aligned coordinates; ``inverse_affine``
    maps aligned to original. Coordinates use PIL's continuous image frame.
    """
    rgb = image.convert("RGB")
    gray = _gray(rgb)
    small, info = _describe(gray, _otsu(gray))
    info["threshold_perturbation_max_angle_degrees"] = 0.
    info["threshold_perturbation_max_center_shift_px256"] = 0.
    info["foreground_retained_fraction"] = 1.
    width, height = rgb.size
    identity = [1., 0., 0., 0., 1., 0.]
    forward = identity.copy()
    inverse = identity.copy()
    if info["geometry_available"]:
        variants = [_describe(gray, info["threshold"] + delta)[1] for delta in (-10, 10)]
        if not all(v["geometry_available"] for v in variants):
            info["threshold_perturbation_max_angle_degrees"] = 180.
            info["threshold_perturbation_max_center_shift_px256"] = 256.
        else:
            info["threshold_perturbation_max_angle_degrees"] = max(
                _angle_difference(info["head_angle_degrees"], v["head_angle_degrees"]) for v in variants)
            info["threshold_perturbation_max_center_shift_px256"] = max(
                float(np.linalg.norm(np.array(v["center_px256"]) - info["center_px256"])) for v in variants)
        ux, uy = info["head_axis"]
        cx = (info["alignment_center_px256"][0] + .5) * width / 256
        cy = (info["alignment_center_px256"][1] + .5) * height / 256
        forward = [-uy, ux, width / 2 + uy * cx - ux * cy,
                   -ux, -uy, height / 2 + ux * cx + uy * cy]
        inverse = [-uy, -ux, cx + uy * width / 2 + ux * height / 2,
                   ux, -uy, cy - ux * width / 2 + uy * height / 2]
        yy, xx = np.nonzero(small)
        source_x = (xx + .5) * width / 256
        source_y = (yy + .5) * height / 256
        tx = forward[0] * source_x + forward[1] * source_y + forward[2]
        ty = forward[3] * source_x + forward[4] * source_y + forward[5]
        info["foreground_retained_fraction"] = float(np.mean((tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)))
    failures = _quality_failures(info)
    # Geometry is estimated in a square 256 frame, matching the square MVTec
    # screw images. Do not silently use its direction on stretched rectangles.
    if width != height:
        failures.append("unsupported_non_square_geometry")
    arr = np.asarray(rgb)
    edge = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]), axis=0)
    fill = np.median(edge, axis=0).astype(np.uint8).tolist()
    fallback = bool(failures)
    if fallback:
        forward = identity.copy()
        inverse = identity.copy()
        aligned = rgb.copy()
    else:
        aligned = rgb.transform(rgb.size, Image.Transform.AFFINE, inverse,
                                resample=Image.Resampling.BILINEAR, fillcolor=tuple(fill))
    metadata = {"version": GEOMETRY_VERSION, "source_size": list(rgb.size),
                "forward_affine": forward, "inverse_affine": inverse,
                "fill_color": fill, "quality_rule": QUALITY_RULE.copy(),
                "quality": {"passed": not fallback, "failures": failures, **info},
                "fallback": fallback, "fallback_policy": "identity",
                "head_direction": "up" if not fallback else "unmodified"}
    return aligned, metadata


def warp_mask(mask, metadata):
    """Warp original-size binary mask to aligned frame with nearest sampling."""
    values = np.asarray(mask)
    width, height = metadata["source_size"]
    if values.ndim != 2 or values.shape != (height, width):
        raise ValueError("mask must have the original image height and width")
    source = Image.fromarray((values != 0).astype(np.uint8) * 255)
    result = source.transform((width, height), Image.Transform.AFFINE,
                              metadata["inverse_affine"], resample=Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(result) > 0


def inverse_heatmap(heatmap, metadata):
    """Map aligned 224x224 scores back to original-frame 224x224 coordinates.

    Bilinear interpolation is used; original pixels whose mapped position lies
    outside the aligned canvas receive zero. For pixel metrics, separately
    transform an all-ones map and restrict evaluation to valid pixels.
    """
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != (224, 224) or not np.isfinite(values).all():
        raise ValueError("heatmap must be finite with shape (224, 224)")
    width, height = metadata["source_size"]
    a, b, c, d, e, f = metadata["forward_affine"]
    coefficients = [a, b * height / width, c * 224 / width,
                    d * width / height, e, f * 224 / height]
    source = Image.fromarray(values)
    result = source.transform((224, 224), Image.Transform.AFFINE, coefficients,
                              resample=Image.Resampling.BILINEAR, fillcolor=0.)
    return np.asarray(result, dtype=np.float32).copy()
