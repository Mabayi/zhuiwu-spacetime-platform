# -*- coding: utf-8 -*-
"""Automatic edge-supported gap recovery for slope-line extraction.

The module deliberately does not consume manual guide polylines.  It searches
nearby same-type slope lines, routes a path through an independently derived
step-edge evidence map, and exports only deduplicated, evidence-filtered bridge
segments.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

from slopeline_enhanced import _direct_step_edges


def _resample(points: np.ndarray, step_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)[:, :2]
    if len(points) < 2:
        return points
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-9:
        return points[:1]
    positions = np.arange(0.0, cumulative[-1] + 1e-9, max(step_m, 1e-3))
    output = []
    for position in positions:
        index = min(
            len(lengths) - 1,
            max(0, int(np.searchsorted(cumulative, position, side="right") - 1)),
        )
        fraction = (position - cumulative[index]) / max(lengths[index], 1e-9)
        output.append(points[index] + fraction * segments[index])
    return np.asarray(output, dtype=float)


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _deduplicate(points: np.ndarray, tolerance_m: float = 0.15) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    kept = [points[0]]
    tolerance = max(float(tolerance_m), 1e-6)
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) >= tolerance:
            kept.append(point)
    if float(np.linalg.norm(points[-1] - kept[-1])) > 1e-9:
        kept.append(points[-1])
    return np.asarray(kept, dtype=float)


def _line_distance_map(
    lines: list[dict],
    line_type: str,
    grid: Any,
    shape: tuple[int, int],
    res: float,
) -> np.ndarray:
    """Return distance to existing lines of one type."""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    dense_step = max(0.5, res * 0.5)
    for line in lines:
        if line.get("type") != line_type:
            continue
        points = np.asarray(line.get("pts", []), dtype=float)
        if points.ndim != 2 or len(points) < 2:
            continue
        points = _resample(points, dense_step)
        rows = np.clip(
            ((grid.ymax - points[:, 1]) / res).astype(int), 0, ny - 1
        )
        cols = np.clip(
            ((points[:, 0] - grid.xmin) / res).astype(int), 0, nx - 1
        )
        mask[rows, cols] = True
    return ndi.distance_transform_edt(~mask) * res


def _score_candidate(candidate: dict) -> float:
    """Ranking score, not a probability or accuracy metric."""
    support = float(candidate["edge_support_2m"])
    novelty = float(candidate["path_novelty_fraction"])
    ratio = float(candidate["length_ratio"])
    ratio_quality = 1.0 - min(max(ratio - 1.0, 0.0) / 0.7, 1.0)
    return float(0.55 * support + 0.35 * novelty + 0.10 * ratio_quality)


def _nms(candidates: list[dict], radius_m: float, max_bridges: int) -> list[dict]:
    """Keep the strongest bridge per local neighborhood and line pair."""
    ordered = sorted(candidates, key=_score_candidate, reverse=True)
    kept: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()
    radius2 = max(0.0, float(radius_m)) ** 2
    for candidate in ordered:
        pair = tuple(sorted((int(candidate["first_line"]), int(candidate["second_line"]))))
        if pair in used_pairs:
            continue
        midpoint = np.asarray(candidate["midpoint"], dtype=float)
        duplicate = False
        for existing in kept:
            if existing["line_type"] != candidate["line_type"]:
                continue
            if float(np.sum((midpoint - existing["midpoint"]) ** 2)) < radius2:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        used_pairs.add(pair)
        if len(kept) >= max_bridges:
            break
    return kept


def recover_auto_gap_bridges(
    lines: list[dict],
    dsm: np.ndarray,
    grid: Any,
    cfg: Any,
    verbose: bool = True,
) -> list[dict]:
    """Recover missing edge segments without using manual guide coordinates."""
    if not bool(getattr(cfg, "auto_gap_enabled", False)) or len(lines) < 2:
        return lines

    res = float(grid.res)
    ny, nx = dsm.shape
    valid = np.isfinite(dsm)
    if not valid.any():
        return lines

    _, fill_indices = ndi.distance_transform_edt(
        ~valid, return_indices=True, return_distances=True
    )
    fill = dsm[tuple(fill_indices)]

    crest_edge, toe_edge = _direct_step_edges(
        dsm,
        res,
        cfg.ray_m,
        float(getattr(cfg, "auto_gap_edge_drop_m", 1.5)),
        float(getattr(cfg, "auto_gap_edge_center_slope_deg", 32.0)),
    )
    edge_dist = {
        "crest": ndi.distance_transform_edt(~crest_edge) * res,
        "toe": ndi.distance_transform_edt(~toe_edge) * res,
    }
    line_dist = {
        line_type: _line_distance_map(lines, line_type, grid, dsm.shape, res)
        for line_type in ("crest", "toe")
    }

    step_m = max(1.0, float(getattr(cfg, "auto_gap_sample_step_m", 5.0)))
    max_gap = max(10.0, float(getattr(cfg, "auto_gap_max_m", 120.0)))
    min_gap = max(2.0, float(getattr(cfg, "auto_gap_min_m", 25.0)))

    samples: list[np.ndarray] = []
    valid_line_ids: list[int] = []
    for line_index, line in enumerate(lines):
        points = np.asarray(line.get("pts", []), dtype=float)
        if points.ndim != 2 or len(points) < 2:
            continue
        samples.append(_resample(points, step_m))
        valid_line_ids.append(line_index)
    if len(samples) < 2:
        return lines

    sample_line_ids = np.concatenate(
        [np.full(len(points), line_index, dtype=int) for line_index, points in zip(valid_line_ids, samples)]
    )
    all_samples = np.vstack(samples)
    trees = {line_index: cKDTree(points) for line_index, points in zip(valid_line_ids, samples)}
    global_tree = cKDTree(all_samples)

    proximity_pairs: set[tuple[int, int]] = set()
    for sample_index, neighbors in enumerate(global_tree.query_ball_point(all_samples, max_gap)):
        first_line = int(sample_line_ids[sample_index])
        for neighbor in neighbors:
            second_line = int(sample_line_ids[neighbor])
            if first_line == second_line:
                continue
            if lines[first_line].get("type") != lines[second_line].get("type"):
                continue
            proximity_pairs.add(tuple(sorted((first_line, second_line))))

    def local_minima(first_line: int, second_line: int) -> list[tuple[np.ndarray, np.ndarray]]:
        first_points = samples[valid_line_ids.index(first_line)]
        second_points = samples[valid_line_ids.index(second_line)]
        if len(first_points) == 0 or len(second_points) == 0:
            return []
        distances, target_indices = trees[second_line].query(first_points, k=1)
        distances = ndi.uniform_filter1d(distances.astype(float), size=3, mode="nearest")
        indices = []
        for index, distance in enumerate(distances):
            if distance > max_gap:
                continue
            left = distances[max(0, index - 1)]
            right = distances[min(len(distances) - 1, index + 1)]
            if distance <= left + 1e-6 and distance <= right + 1e-6:
                indices.append(index)
        if len(distances) and distances[0] <= max_gap:
            indices.append(0)
        if len(distances) and distances[-1] <= max_gap:
            indices.append(len(distances) - 1)
        selected = []
        separation = max(1, int(round(20.0 / step_m)))
        for index in sorted(set(indices), key=lambda value: distances[value]):
            if all(abs(index - existing) >= separation for existing in selected):
                selected.append(index)
            if len(selected) >= 8:
                break
        return [
            (first_points[index], second_points[int(target_indices[index])])
            for index in sorted(selected)
        ]

    raw_candidates = []
    for first_line, second_line in sorted(proximity_pairs):
        pair = local_minima(first_line, second_line) + local_minima(second_line, first_line)
        unique = {}
        for first_point, second_point in pair:
            midpoint = 0.5 * (first_point + second_point)
            key = (
                round(float(midpoint[0]) / 10.0),
                round(float(midpoint[1]) / 10.0),
                round(float(np.linalg.norm(first_point - second_point)) / 10.0),
            )
            unique.setdefault(key, (first_point, second_point))
        for first_point, second_point in unique.values():
            gap = float(np.linalg.norm(second_point - first_point))
            if min_gap <= gap <= max_gap:
                raw_candidates.append((first_line, second_line, first_point, second_point, gap))

    def xy_to_rc(point: np.ndarray) -> tuple[int, int]:
        row = int(np.clip((grid.ymax - point[1]) / res, 0, ny - 1))
        col = int(np.clip((point[0] - grid.xmin) / res, 0, nx - 1))
        return row, col

    quick_candidates = []
    for first_line, second_line, first_point, second_point, gap in raw_candidates:
        line_type = str(lines[first_line]["type"])
        fractions = np.linspace(0.0, 1.0, 31)
        chord = first_point[None, :] * (1.0 - fractions[:, None]) + second_point[None, :] * fractions[:, None]
        rows = np.clip(((grid.ymax - chord[:, 1]) / res).astype(int), 0, ny - 1)
        cols = np.clip(((chord[:, 0] - grid.xmin) / res).astype(int), 0, nx - 1)
        edge_values = edge_dist[line_type][rows, cols]
        line_values = line_dist[line_type][rows, cols]
        median_edge = float(np.median(edge_values))
        fraction_edge_4m = float(np.mean(edge_values <= 4.0))
        novelty_fraction = float(np.mean(line_values >= 3.0))
        if median_edge > 8.0 or fraction_edge_4m < 0.35 or novelty_fraction < 0.20:
            continue
        quick_score = gap + 3.0 * median_edge - 20.0 * fraction_edge_4m - 10.0 * novelty_fraction
        quick_candidates.append({
            "quick_score": float(quick_score),
            "first_line": first_line,
            "second_line": second_line,
            "first_point": first_point,
            "second_point": second_point,
            "gap_m": gap,
        })
    quick_candidates.sort(key=lambda item: item["quick_score"])
    quick_candidates = quick_candidates[
        : max(1, int(getattr(cfg, "auto_gap_max_path_candidates", 2000)))
    ]

    support_min = float(getattr(cfg, "auto_gap_edge_support_2m_min", 0.75))
    novelty_min = float(getattr(cfg, "auto_gap_novelty_min", 0.75))
    ratio_max = float(getattr(cfg, "auto_gap_length_ratio_max", 1.50))
    weak_support_min = float(getattr(cfg, "auto_gap_weak_edge_support_2m_min", 0.70))
    weak_novelty_min = float(getattr(cfg, "auto_gap_weak_novelty_min", 0.65))

    routed = []
    for candidate in quick_candidates:
        first_line = int(candidate["first_line"])
        second_line = int(candidate["second_line"])
        first_point = np.asarray(candidate["first_point"], dtype=float)
        second_point = np.asarray(candidate["second_point"], dtype=float)
        gap = float(candidate["gap_m"])
        line_type = str(lines[first_line]["type"])
        first_rc = xy_to_rc(first_point)
        second_rc = xy_to_rc(second_point)
        padding = int(20.0 / res) + 4
        row0 = max(0, min(first_rc[0], second_rc[0]) - padding)
        row1 = min(ny - 1, max(first_rc[0], second_rc[0]) + padding)
        col0 = max(0, min(first_rc[1], second_rc[1]) - padding)
        col1 = min(nx - 1, max(first_rc[1], second_rc[1]) + padding)
        sub_edge = edge_dist[line_type][row0 : row1 + 1, col0 : col1 + 1]
        sub_line = line_dist[line_type][row0 : row1 + 1, col0 : col1 + 1]
        sub_z = fill[row0 : row1 + 1, col0 : col1 + 1]
        height, width = sub_edge.shape
        yy, xx = np.mgrid[0:height, 0:width]
        pixel_x = grid.xmin + (xx + col0 + 0.5) * res
        pixel_y = grid.ymax - (yy + row0 + 0.5) * res
        vector_x = second_point[0] - first_point[0]
        vector_y = second_point[1] - first_point[1]
        denominator = max(vector_x * vector_x + vector_y * vector_y, 1e-9)
        fraction = np.clip(
            ((pixel_x - first_point[0]) * vector_x + (pixel_y - first_point[1]) * vector_y)
            / denominator,
            0.0,
            1.0,
        )
        lateral = (
            np.abs(
                (pixel_x - first_point[0]) * vector_y
                - (pixel_y - first_point[1]) * vector_x
            )
            / np.sqrt(denominator)
        )
        z_linear = fill[first_rc] + fraction * (fill[second_rc] - fill[first_rc])
        cost = 1.0 + 4.0 * sub_edge + 0.03 * lateral + 0.5 * np.abs(sub_z - z_linear)
        cost[~np.isfinite(sub_z)] = 25.0
        try:
            path, _ = route_through_array(
                cost,
                (first_rc[0] - row0, first_rc[1] - col0),
                (second_rc[0] - row0, second_rc[1] - col0),
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            continue
        edge_values = np.asarray([sub_edge[row, col] for row, col in path], dtype=float)
        line_values = np.asarray([sub_line[row, col] for row, col in path], dtype=float)
        path_points = np.asarray([
            [
                grid.xmin + (col + col0 + 0.5) * res,
                grid.ymax - (row + row0 + 0.5) * res,
            ]
            for row, col in path
        ], dtype=float)
        path_length = _length(path_points)
        if path_length <= 1e-9:
            continue
        support_2m = float(np.mean(edge_values <= 2.0))
        novelty_fraction = float(np.mean(line_values >= 3.0))
        length_ratio = path_length / max(gap, 1e-9)
        strict = (
            support_2m >= support_min
            and novelty_fraction >= novelty_min
            and length_ratio <= ratio_max
        )
        weak = (
            support_2m >= weak_support_min
            and novelty_fraction >= weak_novelty_min
            and length_ratio <= max(ratio_max, 1.55)
        )
        if not (strict or weak):
            continue
        routed.append({
            "first_line": first_line,
            "second_line": second_line,
            "first_point": first_point,
            "second_point": second_point,
            "midpoint": 0.5 * (first_point + second_point),
            "gap_m": gap,
            "path_points": path_points,
            "path_length_m": path_length,
            "edge_support_2m": support_2m,
            "path_novelty_fraction": novelty_fraction,
            "length_ratio": length_ratio,
            "line_type": line_type,
            "evidence_class": "strong" if strict else "weak",
        })

    selected = _nms(
        routed,
        radius_m=float(getattr(cfg, "auto_gap_nms_radius_m", 20.0)),
        max_bridges=max(1, int(getattr(cfg, "auto_gap_max_bridges", 60))),
    )
    output = list(lines)
    for candidate in selected:
        connector = _deduplicate(
            np.vstack([
                np.asarray(candidate["first_point"], dtype=float)[None, :],
                np.asarray(candidate["path_points"], dtype=float),
                np.asarray(candidate["second_point"], dtype=float)[None, :],
            ]),
            tolerance_m=max(0.10, res * 0.25),
        )
        if len(connector) < 2:
            continue
        first_rc = xy_to_rc(np.asarray(candidate["first_point"], dtype=float))
        second_rc = xy_to_rc(np.asarray(candidate["second_point"], dtype=float))
        z_values = [float(fill[first_rc]), float(fill[second_rc])]
        z_value = float(np.nanmean(z_values)) if np.isfinite(z_values).any() else 0.0
        score = _score_candidate(candidate)
        output.append({
            "type": candidate["line_type"],
            "pts": connector,
            "length": _length(connector),
            "z": z_value,
            "_source": "auto_gap_bridge",
            "_link_mode": "auto_edge_path",
            "_confidence": float(np.clip(score, 0.0, 1.0)),
            "_evidence_class": candidate["evidence_class"],
            "_anchor_line_start": int(candidate["first_line"]),
            "_anchor_line_end": int(candidate["second_line"]),
            "_anchor_gap_m": float(candidate["gap_m"]),
            "_path_length_m": float(candidate["path_length_m"]),
            "_edge_support_2m": float(candidate["edge_support_2m"]),
            "_path_novelty_fraction": float(candidate["path_novelty_fraction"]),
            "_length_ratio": float(candidate["length_ratio"]),
            "_auto_gap_method": "minimum_cost_step_edge_path",
        })

    if verbose:
        strong = sum(1 for item in selected if item["evidence_class"] == "strong")
        weak = len(selected) - strong
        print(
            "[auto-gap] "
            f"pairs={len(proximity_pairs)} raw={len(raw_candidates)} "
            f"routed={len(routed)} selected={len(selected)} "
            f"strong={strong} weak={weak}"
        )
    return output
