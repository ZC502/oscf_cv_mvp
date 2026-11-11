#!/usr/bin/env python3
"""OSCF-CV MVP: vision-model-neutral spatial constraint auditing.

This is a research prototype, not a physical-causality inference system.
Runtime dependencies: numpy, Pillow. Optional: ultralytics for YOLO/SAM2.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any
import math

import numpy as np
from PIL import Image, ImageDraw

EPS = 1e-10

# ---- Standard Cayley-Dickson octonions. The first four components are H. ----
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_conj(a: np.ndarray) -> np.ndarray:
    return np.array([a[0], -a[1], -a[2], -a[3]], dtype=np.float64)


def oct_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cayley-Dickson: (a,b)(c,d) = (ac - conj(d)b, da + b conj(c))."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != (8,) or y.shape != (8,):
        raise ValueError("Octonions must have exactly 8 components")
    a, b = x[:4], x[4:]
    c, d = y[:4], y[4:]
    return np.concatenate((quat_mul(a, c) - quat_mul(quat_conj(d), b),
                           quat_mul(d, a) + quat_mul(b, quat_conj(c))))


def associator(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return oct_mul(oct_mul(a, b), c) - oct_mul(a, oct_mul(b, c))


def normalized_associator(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
    vec = associator(a, b, c)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)), EPS)
    return float(np.linalg.norm(vec) / denom), vec / denom


@dataclass
class Observation:
    id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy pixel coordinates
    mask: np.ndarray | None = None
    depth: float | None = None  # calibrated/relative depth already in [0,1]
    semantic_weight: float = 0.5  # application policy value, not a class embedding

    def __post_init__(self) -> None:
        self.bbox = tuple(float(v) for v in self.bbox)
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError(f"invalid bbox for {self.id}: {self.bbox}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.depth is not None and not 0 <= self.depth <= 1:
            raise ValueError("depth must be normalized to [0, 1]")
        if not 0 <= self.semantic_weight <= 1:
            raise ValueError("semantic_weight must be normalized to [0, 1]")


def mask_major_axis_proxy(mask: np.ndarray | None) -> tuple[float, bool]:
    """2-D principal-axis orientation proxy, NOT a surface normal."""
    if mask is None:
        return 0.0, False
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return 0.0, False
    covariance = np.cov(np.stack((xs, ys), axis=0))
    vals, vecs = np.linalg.eigh(covariance)
    if vals[-1] <= EPS or vals[-1] / max(vals[0], EPS) < 1.25:
        return 0.0, False  # orientation ambiguous for near-round objects
    dominant = vecs[:, -1]
    theta = math.atan2(dominant[1], dominant[0])
    return float(math.sin(2.0 * theta)), True  # invariant to 180-degree flips


def proximity_coupling(obs: Observation, observations: list[Observation], size: tuple[int, int]) -> float:
    if len(observations) < 2:
        return 0.0
    w, h = size
    cx = (obs.bbox[0] + obs.bbox[2]) / 2.0
    cy = (obs.bbox[1] + obs.bbox[3]) / 2.0
    diag = math.hypot(w, h)
    values = []
    for other in observations:
        if other.id == obs.id:
            continue
        ox = (other.bbox[0] + other.bbox[2]) / 2.0
        oy = (other.bbox[1] + other.bbox[3]) / 2.0
        values.append(math.exp(-3.0 * math.hypot(cx - ox, cy - oy) / diag))
    return float(max(values, default=0.0))


def encode_scene(observations: list[Observation], size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    """8-dimensional, dimensionless octonions plus validity/semantic metadata."""
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("image size must be positive")
    if len({o.id for o in observations}) != len(observations):
        raise ValueError("duplicate observation IDs")
    result: dict[str, dict[str, Any]] = {}
    for ob in observations:
        x1, y1, x2, y2 = ob.bbox
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        if ob.mask is not None and ob.mask.shape != (h, w):
            raise ValueError(f"mask shape for {ob.id} does not match {(h, w)}")
        area = (float(np.count_nonzero(ob.mask)) if ob.mask is not None
                else (x2 - x1) * (y2 - y1))
        scale = float(np.sqrt(area / (w * h)))
        axis, axis_valid = mask_major_axis_proxy(ob.mask)
        coupling = proximity_coupling(ob, observations, size)
        q = np.array([ob.confidence, cx, cy,
                      ob.depth if ob.depth is not None else 0.0,
                      scale, axis, ob.semantic_weight, coupling], dtype=np.float64)
        result[ob.id] = {
            "label": ob.label,
            "bbox_xyxy": list(ob.bbox),
            "q": q,
            "depth_valid": ob.depth is not None,
            "axis_proxy_valid": axis_valid,
            "has_mask": ob.mask is not None,
        }
    return result


def find_observation(observations: list[Observation], label: str) -> Observation | None:
    return max((ob for ob in observations if ob.label == label),
               key=lambda ob: ob.confidence, default=None)


def support_rule(cup: Observation, support: Observation, tolerance_px: float = 14.0) -> dict[str, Any]:
    """Declared 2-D 'on top of' contract. It is NOT evidence of true contact in 3-D."""
    x1, y1, x2, y2 = cup.bbox
    sx1, sy1, sx2, sy2 = support.bbox
    bottom_gap = float(y2 - sy1)
    overlap = max(0.0, min(x2, sx2) - max(x1, sx1))
    horizontal_fraction = overlap / max(x2 - x1, EPS)
    valid = abs(bottom_gap) <= tolerance_px and horizontal_fraction >= 0.5
    return {
        "status": "PASS" if valid else "WARN",
        "vertical_gap_px": round(bottom_gap, 3),
        "horizontal_overlap_fraction": round(horizontal_fraction, 4),
        "tolerance_px": tolerance_px,
        "note": "User-declared 2-D support relation (not a proven causal contact)",
    }


class MaskProjectionError(ValueError):
    """Counterfactual mask translation would clip observed mask pixels."""


def translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a full-frame binary mask by the nearest integer pixel offset.

    Bounding boxes retain their subpixel translation. Nearest-pixel translation
    preserves the mask's area and shape when no foreground pixel is clipped.
    Clipping is rejected rather than silently changing the octonion's area or
    orientation features and introducing a spurious associator residual.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D full-frame array")
    shift_x, shift_y = int(round(dx)), int(round(dy))
    if shift_x == 0 and shift_y == 0:
        return mask.copy()
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    shifted = np.zeros((height, width), dtype=bool)
    if xs.size == 0:
        return shifted
    new_xs, new_ys = xs + shift_x, ys + shift_y
    if (new_xs.min() < 0 or new_xs.max() >= width or
            new_ys.min() < 0 or new_ys.max() >= height):
        raise MaskProjectionError(
            f"mask would be clipped by translation dx={dx:.3f}, dy={dy:.3f} "
            f"(rounded to {shift_x}, {shift_y} pixels)"
        )
    shifted[new_ys, new_xs] = True
    return shifted


def project_onto_support(cup: Observation, support: Observation) -> Observation:
    """Slide the object onto its declared support, translating any mask too.

    When no motion is required, the original observation (including its mask)
    is unchanged. A counterfactual that would clip a mask is rejected so the
    associator feature can be reported as unavailable, not as a false anomaly.
    """
    x1, y1, x2, y2 = cup.bbox
    sx1, sy1, sx2, _ = support.bbox
    dy = sy1 - y2
    width = x2 - x1
    cx = (x1 + x2) / 2.0
    xmin = sx1 + width / 2.0
    xmax = sx2 - width / 2.0
    new_cx = (sx1 + sx2) / 2.0 if xmin > xmax else min(max(cx, xmin), xmax)
    dx = new_cx - cx
    if abs(dx) < EPS and abs(dy) < EPS:
        return cup
    projected_mask = (translate_mask(cup.mask, dx, dy)
                      if cup.mask is not None else None)
    return replace(cup,
                   bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                   mask=projected_mask)


def rule_associator_features(
        observations: list[Observation], size: tuple[int, int],
        target: Observation, support: Observation) -> dict[str, Any]:
    """Compare associator before vs after a declared geometric intervention.

    The difference is an algebraic *feature*. Without controls/ground truth it
    does not establish physical causation or extra predictive power.
    """
    third = next((x for x in observations if x.id not in (target.id, support.id)), None)
    if third is None:
        return {"status": "INSUFFICIENT", "reason": "need a third distinct entity for an associator"}
    before = encode_scene(observations, size)
    try:
        updated = [project_onto_support(x, support) if x.id == target.id else x
                   for x in observations]
    except MaskProjectionError as exc:
        return {"status": "INSUFFICIENT", "reason": str(exc)}
    after = encode_scene(updated, size)
    before_vals = [before[x.id]["q"] for x in (target, support, third)]
    after_vals = [after[x.id]["q"] for x in (target, support, third)]
    score_obs, assoc_obs = normalized_associator(*before_vals)
    score_cf, assoc_cf = normalized_associator(*after_vals)
    # Geometrically changed one 8-D code: competing linear baseline.
    q_change = float(np.linalg.norm(before_vals[0] - after_vals[0]))
    return {
        "status": "MEASURED",
        "triple": [target.id, support.id, third.id],
        "associator_norm_observed": round(score_obs, 8),
        "associator_norm_projected": round(score_cf, 8),
        "associator_counterfactual_delta": round(float(np.linalg.norm(assoc_obs - assoc_cf)), 8),
        "linear_8d_counterfactual_delta": round(q_change, 8),
        "note": "Proxy geometry intervention, not an identified causal effect",
    }


def audit_scene(observations: list[Observation], size: tuple[int, int],
                contracts: list[tuple[str, str]]) -> dict[str, Any]:
    codes = encode_scene(observations, size)
    findings = []
    for target_label, support_label in contracts:
        target = find_observation(observations, target_label)
        support = find_observation(observations, support_label)
        if target is None or support is None or target.id == support.id:
            findings.append({"contract": f"{target_label}:on:{support_label}",
                             "status": "INSUFFICIENT", "reason": "required pair not found"})
            continue
        result = support_rule(target, support)
        algebra = rule_associator_features(observations, size, target, support)
        findings.append({"contract": f"{target_label}:on:{support_label}",
                         "target_id": target.id, "support_id": support.id, **result,
                         "octonion_features": algebra})
    statuses = [x["status"] for x in findings]
    decision = ("WARN" if "WARN" in statuses else
                "INSUFFICIENT" if "INSUFFICIENT" in statuses else
                "PASS" if findings else "UNCONSTRAINED")
    return {
        "decision": decision,
        "image_size": {"width": size[0], "height": size[1]},
        "representation": "Cayley-Dickson octonion (8D), all channels normalized/proxied",
        "q_channel_schema": ["confidence", "x_center_norm", "y_center_norm", "depth_norm_or_zero",
                             "sqrt_area_fraction", "mask_axis_proxy_or_zero",
                             "semantic_policy_weight", "proximity_coupling"],
        "objects": [{"id": ob.id, "label": ob.label,
                     "q": [round(float(v), 7) for v in codes[ob.id]["q"]],
                     "bbox_xyxy": list(ob.bbox),
                     "depth_valid": codes[ob.id]["depth_valid"],
                     "axis_proxy_valid": codes[ob.id]["axis_proxy_valid"],
                     "has_mask": codes[ob.id]["has_mask"]}
                    for ob in observations],
        "findings": findings,
        "limitations": [
            "Single-image 2D support is a declared geometric prior, not a causal conclusion.",
            "Associator residuals are telemetry; no decision is based solely on a nonzero associator.",
            "Missing depth or mask stays explicitly unknown, never silently inferred.",
        ],
    }


class TemporalCache:
    """Simple frame-to-frame state cache. Requires stable object IDs from a tracker."""
    def __init__(self) -> None:
        self._prev: dict[str, tuple[int, np.ndarray]] = {}

    def update(self, frame_idx: int, encoded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        changes = []
        next_prev: dict[str, tuple[int, np.ndarray]] = {}
        for oid, obj in encoded.items():
            q = obj["q"]
            if oid in self._prev:
                prev_frame, prev = self._prev[oid]
                if frame_idx > prev_frame:
                    changes.append({
                        "id": oid,
                        "elapsed_frames": frame_idx - prev_frame,
                        "xy_displacement_norm": float(np.linalg.norm(q[1:3] - prev[1:3])),
                        "delta_8d_l2": float(np.linalg.norm(q - prev)),
                    })
            next_prev[oid] = (frame_idx, q.copy())
        self._prev = next_prev
        return changes


def observations_from_json(path: Path) -> list[Observation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise ValueError('detections JSON must have an "objects" array')
    return [Observation(
        id=str(x.get("id", f"obj{i}")), label=str(x["label"]),
        confidence=float(x.get("confidence", x.get("conf", 1.0))),
        bbox=tuple(x["bbox"]),
        depth=(float(x["depth"]) if "depth" in x else None),
        semantic_weight=float(x.get("semantic_weight", 0.5)),
    ) for i, x in enumerate(payload["objects"])]


def add_depth_from_npy(observations: list[Observation], depth_file: Path, size: tuple[int, int]) -> list[Observation]:
    depth_map = np.load(depth_file)
    w, h = size
    if depth_map.shape != (h, w) or not np.all(np.isfinite(depth_map)):
        raise ValueError("depth map must be a finite HxW numpy array")
    if depth_map.min() < 0 or depth_map.max() > 1:
        raise ValueError("depth map must be normalized to [0,1] by the caller")
    result = []
    for o in observations:
        x1, y1, x2, y2 = o.bbox
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(w, math.ceil(x2)), min(h, math.ceil(y2))
        region = depth_map[top:bottom, left:right]
        if o.mask is not None and region.size:
            m = o.mask[top:bottom, left:right].astype(bool)
            samples = region[m]
        else:
            samples = region.ravel()
        value = float(np.median(samples)) if samples.size else None
        result.append(replace(o, depth=value))
    return result


def observations_from_ultralytics(image_path: Path, yolo_weights: str,
                                   sam_weights: str | None, min_conf: float) -> list[Observation]:
    """Lazy-load optional YOLO and SAM/SAM2 model adapters.

    SAM gets one box at a time to avoid ambiguous mask/detection ordering.
    Class labels/confidence always come from YOLO, not SAM mask quality.
    """
    try:
        from ultralytics import YOLO, SAM
    except ImportError as exc:
        raise RuntimeError("Install 'ultralytics' for model inference: pip install ultralytics") from exc
    yolo = YOLO(yolo_weights)
    result = yolo.predict(source=str(image_path), conf=min_conf, verbose=False)[0]
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    sam = SAM(sam_weights) if sam_weights else None
    width, height = Image.open(image_path).size
    observations: list[Observation] = []
    for idx, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
        label = str(result.names[int(cls)])
        mask = None
        if sam is not None:
            sr = sam.predict(source=str(image_path), bboxes=box.astype(float).tolist(),
                             verbose=False)[0]
            if sr.masks is not None and len(sr.masks.data):
                arr = sr.masks.data[0].cpu().numpy().astype(np.uint8)
                if arr.shape != (height, width):
                    arr = np.array(Image.fromarray(arr).resize((width, height),
                                                                resample=Image.Resampling.NEAREST))
                mask = arr.astype(bool)
        observations.append(Observation(
            id=f"{label}_{idx}", label=label, confidence=float(conf),
            bbox=tuple(float(v) for v in box), mask=mask))
    return observations


def draw_audit(image: Image.Image, observations: list[Observation], report: dict[str, Any], path: Path) -> None:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    warn_ids = {f.get("target_id") for f in report["findings"] if f["status"] == "WARN"}
    for ob in observations:
        coords = tuple(int(v) for v in ob.bbox)
        color = (210, 48, 48) if ob.id in warn_ids else (32, 146, 90)
        d.rectangle(coords, outline=color, width=3)
        d.text((coords[0] + 3, max(2, coords[1] - 14)),
               f"{ob.label} {ob.confidence:.2f}", fill=color)
    d.rectangle((0, 0, min(img.width, 390), 27), fill=(23, 36, 48))
    d.text((10, 8), f"OSCF-CV: {report['decision']}", fill=(255, 255, 255))
    img.save(path)


def save_audit(image: Image.Image, observations: list[Observation],
               contracts: list[tuple[str, str]], out: Path, stem: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    report = audit_scene(observations, image.size, contracts)
    report["input_kind"] = "synthetic_mock" if stem.startswith("demo_") else "image"
    (out / f"{stem}_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    draw_audit(image, observations, report, out / f"{stem}_annotated.png")
    return report


def create_synthetic_scene(out: Path, floating_cup: bool) -> tuple[Image.Image, list[Observation]]:
    """Generated geometry with known declared contract. No CV weights are involved."""
    image = Image.new("RGB", (640, 420), (238, 244, 246))
    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 270, 550, 322), fill=(111, 80, 59))
    draw.rectangle((120, 322, 150, 410), fill=(85, 63, 50))
    draw.rectangle((490, 322, 520, 410), fill=(85, 63, 50))
    cup_box = (285, 105, 335, 155) if floating_cup else (285, 220, 335, 270)
    draw.rounded_rectangle(cup_box, radius=5, fill=(45, 129, 181))
    draw.rectangle((155, 246, 245, 270), fill=(229, 171, 54))
    data = [
        Observation("cup_0", "cup", 0.94, cup_box, semantic_weight=0.7),
        Observation("table_0", "table", 0.97, (90, 270, 550, 322), semantic_weight=0.8),
        Observation("book_0", "book", 0.91, (155, 246, 245, 270), semantic_weight=0.6),
    ]
    return image, data


def run_demo(out: Path) -> None:
    for name, floating in (("demo_supported", False), ("demo_floating", True)):
        im, observations = create_synthetic_scene(out, floating)
        out.mkdir(parents=True, exist_ok=True)
        im.save(out / f"{name}_input.png")
        report = save_audit(im, observations, [("cup", "table")], out, name)
        finding = report["findings"][0]
        print(name, report["decision"],
              "vertical_gap_px=", finding["vertical_gap_px"],
              "associator_cf_delta=", finding["octonion_features"].get("associator_counterfactual_delta"))
    print(f"Wrote demo images and reports under {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="run 2 deterministic synthetic image scenarios")
    mode.add_argument("--image", type=Path, help="path to real image")
    parser.add_argument("--detections-json", type=Path,
                        help='optional existing detections {"objects": [...]} (no model weights)')
    parser.add_argument("--yolo-model", default="yolo26n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--sam-model", default=None,
                        help="optional SAM weights, e.g. sam2.1_t.pt")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--depth-npy", type=Path, help="optional normalized HxW relative-depth .npy")
    parser.add_argument("--on", action="append", default=[], metavar="TARGET:SUPPORT",
                        help="declare 2D support relation, e.g. cup:table (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.demo:
        run_demo(args.out)
        return
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    im = Image.open(args.image)
    if args.detections_json:
        observations = observations_from_json(args.detections_json)
    else:
        observations = observations_from_ultralytics(args.image, args.yolo_model,
                                                    args.sam_model, args.min_conf)
    if args.depth_npy:
        observations = add_depth_from_npy(observations, args.depth_npy, im.size)
    contracts = []
    for raw in args.on:
        if raw.count(":") != 1 or not all(part.strip() for part in raw.split(":")):
            parser.error("--on must be TARGET:SUPPORT, such as cup:table")
        target, support = raw.split(":")
        contracts.append((target.strip(), support.strip()))
    report = save_audit(im, observations, contracts, args.out, args.image.stem)
    print(json.dumps({"decision": report["decision"], "count": len(observations),
                      "findings": report["findings"]}, indent=2, ensure_ascii=False))
    print(f"Outputs under {args.out.resolve()}")


if __name__ == "__main__":
    main()
