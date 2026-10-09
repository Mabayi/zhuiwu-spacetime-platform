# -*- coding: utf-8 -*-
"""Enhanced gap recovery for slope-line extraction."""
from __future__ import annotations

import math

import numpy as np
import scipy.ndimage as ndi
from skimage.graph import route_through_array


def _shift_with_nan(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    ny, nx = array.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    rr0 = max(0, -dr)
    rr1 = min(ny, ny - dr)
    cc0 = max(0, -dc)
    cc1 = min(nx, nx - dc)
    if rr0 >= rr1 or cc0 >= cc1:
        return out
    out[rr0:rr1, cc0:cc1] = array[rr0 + dr:rr1 + dr, cc0 + dc:cc1 + dc]
    return out


def _direct_step_edges(
    dsm: np.ndarray,
    res: float,
    ray_m: list[float],
    drop_m: float,
    center_slope_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = dsm.shape
    valid = np.isfinite(dsm)
    gy, gx = np.gradient(dsm, res, res)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    rays = sorted({max(1, int(round(value / res))) for value in ray_m})
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]
    crest_votes = np.zeros((ny, nx), dtype=np.uint8)
    toe_votes = np.zeros((ny, nx), dtype=np.uint8)
    crest_mag = np.zeros((ny, nx), dtype=np.float32)
    toe_mag = np.zeros((ny, nx), dtype=np.float32)

    for dr, dc in directions:
        differences = []
        for radius in rays:
            shifted = _shift_with_nan(dsm, dr * radius, dc * radius)
            differences.append(dsm - shifted)
        stack = np.stack(differences, axis=0)
        with np.errstate(all="ignore"):
            crest_valid = stack > drop_m
            toe_valid = (-stack) > drop_m
            crest_count = np.sum(crest_valid, axis=0).astype(np.uint8)
            toe_count = np.sum(toe_valid, axis=0).astype(np.uint8)
            crest_sum = np.sum(np.where(crest_valid, stack, 0.0), axis=0)
            toe_sum = np.sum(np.where(toe_valid, -stack, 0.0), axis=0)
            crest_mean = np.divide(
                crest_sum, crest_count,
                out=np.zeros_like(crest_sum, dtype=np.float32),
                where=crest_count > 0,
            )
            toe_mean = np.divide(
                toe_sum, toe_count,
                out=np.zeros_like(toe_sum, dtype=np.float32),
                where=toe_count > 0,
            )
        crest_votes = np.maximum(crest_votes, crest_count)
        toe_votes = np.maximum(toe_votes, toe_count)
        crest_mag = np.fmax(crest_mag, np.nan_to_num(crest_mean, nan=0.0))
        toe_mag = np.fmax(toe_mag, np.nan_to_num(toe_mean, nan=0.0))

    required_votes = max(2, int(math.ceil(len(rays) * 0.6)))
    center_ok = valid & np.isfinite(slope_deg) & (slope_deg <= center_slope_deg)
    crest = center_ok & (crest_votes >= required_votes)
    toe = center_ok & (toe_votes >= required_votes)
    both = crest & toe
    crest[both & (toe_mag > crest_mag)] = False
    toe[both & (crest_mag >= toe_mag)] = False
    return crest, toe

def _simplify_path(points: np.ndarray, tolerance_m: float) -> np.ndarray:
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start = points[first]
        end = points[last]
        segment = end - start
        length2 = float(segment @ segment)
        if length2 < 1e-12:
            distance = np.hypot(points[first + 1:last, 0] - start[0], points[first + 1:last, 1] - start[1])
        else:
            fraction = np.clip(((points[first + 1:last] - start) @ segment) / length2, 0, 1)
            projection = start + fraction[:, None] * segment
            distance = np.hypot(points[first + 1:last, 0] - projection[:, 0], points[first + 1:last, 1] - projection[:, 1])
        index = int(np.argmax(distance))
        if distance[index] > tolerance_m:
            keep[first + 1 + index] = True
            stack.append((first, first + 1 + index))
            stack.append((first + 1 + index, last))
    return points[keep]

def _endpoint_records(lines, fill, xy_to_rc):
    records = []
    for line_index, line in enumerate(lines):
        points = line["pts"]
        segment = np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))
        for which in ("s", "e"):
            if which == "s":
                position = points[0]
                target = min(15.0, max(segment[-1] * 0.4, 0.1)) if len(segment) else 0.0
                tangent_index = int(np.searchsorted(segment, target)) + 1 if len(segment) else 0
                tangent_index = min(tangent_index, len(points) - 1)
                tangent = points[0] - points[tangent_index]
            else:
                position = points[-1]
                target = max(segment[-1] - min(15.0, max(segment[-1] * 0.4, 0.1)), 0.0) if len(segment) else 0.0
                tangent_index = int(np.searchsorted(segment, target)) - 1 if len(segment) else len(points) - 1
                tangent_index = max(tangent_index, 0)
                tangent = points[-1] - points[tangent_index]
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                continue
            row, col = xy_to_rc(position[0], position[1])
            z_value = fill[row, col]
            if not np.isfinite(z_value):
                continue
            records.append({
                "line": line_index,
                "which": which,
                "pos": position,
                "tan": tangent / norm,
                "z": float(z_value),
                "rc": (row, col),
            })
    return records


def _mode_rank(mode: str) -> int:
    if mode == "regular_edge":
        return 0
    if mode == "edge_supported_short":
        return 1
    if mode == "edge_supported_long":
        return 2
    if mode == "regular_low_evidence":
        return 3
    return 4

def link_lines_enhanced(lines, dsm, g, cfg, verbose=True):
    """Link fragments with edge support and an inferred-link layer."""
    if cfg.link_gap_m <= 0 or len(lines) < 2:
        return lines
    res = g.res
    ny, nx = dsm.shape
    valid = np.isfinite(dsm)
    if not valid.any():
        return lines
    distance_h, index_h = ndi.distance_transform_edt(~valid, return_indices=True, return_distances=True)
    fill = dsm[tuple(index_h)]
    fill[distance_h > 6] = np.nan
    crest_edge, toe_edge = _direct_step_edges(dsm, res, cfg.ray_m, cfg.min_step_drop_m, cfg.platform_slope_deg + 10.0)
    edge_masks = {"crest": crest_edge, "toe": toe_edge}
    edge_distance = {key: ndi.distance_transform_edt(~mask) for key, mask in edge_masks.items()}

    def xy_to_rc(x, y):
        col = int(np.clip((x - g.xmin) / res, 0, nx - 1))
        row = int(np.clip((g.ymax - y) / res, 0, ny - 1))
        return row, col

    def rc_to_xy(row, col):
        return g.xmin + (col + 0.5) * res, g.ymax - (row + 0.5) * res

    total_merged = 0
    for pass_index in range(cfg.link_passes):
        distance_by_type = {}
        for line_type in ("crest", "toe"):
            mask = np.zeros((ny, nx), dtype=bool)
            for line in lines:
                if line["type"] != line_type:
                    continue
                rows = np.clip(((g.ymax - line["pts"][:, 1]) / res).astype(int), 0, ny - 1)
                cols = np.clip(((line["pts"][:, 0] - g.xmin) / res).astype(int), 0, nx - 1)
                mask[rows, cols] = True
            distance_by_type[line_type] = ndi.distance_transform_edt(~mask) if mask.any() else None
        protected = set()
        if bool(getattr(cfg, "edge_seed_protect_links", False)):
            protected_indices = set(getattr(cfg, "edge_seed_protected_region_indices", []) or [])
            for line_index, line in enumerate(lines):
                if line.get("_source") not in ("edge_seed", "edge_seed_linked"):
                    continue
                region_index = int(line.get("_seed_region_index", -1))
                if not protected_indices or region_index in protected_indices:
                    protected.add(line_index)
        endpoints = _endpoint_records(lines, fill, xy_to_rc)
        candidates = []
        for first_index in range(len(endpoints)):
            for second_index in range(first_index + 1, len(endpoints)):
                first = endpoints[first_index]
                second = endpoints[second_index]
                if first["line"] == second["line"]:
                    continue
                if first["line"] in protected or second["line"] in protected:
                    continue
                if lines[first["line"]]["type"] != lines[second["line"]]["type"]:
                    continue
                delta = second["pos"] - first["pos"]
                gap = float(np.hypot(delta[0], delta[1]))
                if gap < 0.5 or gap > cfg.link_infer_gap_m:
                    continue
                unit = delta / gap
                cos_first = float(first["tan"] @ unit)
                cos_second = float(second["tan"] @ (-unit))
                delta_z = abs(first["z"] - second["z"])
                lateral_first = abs(float(unit[0] * first["tan"][1] - unit[1] * first["tan"][0])) * gap
                lateral_second = abs(float(unit[0] * second["tan"][1] - unit[1] * second["tan"][0])) * gap
                lateral = max(lateral_first, lateral_second)
                regular = gap <= cfg.link_gap_m and delta_z <= cfg.link_dz_max_m and cos_first >= cfg.link_min_cos and cos_second >= cfg.link_min_cos and lateral <= cfg.link_lateral_max_m
                short = gap <= cfg.link_short_gap_m and delta_z <= cfg.link_short_dz_max_m and lateral <= cfg.link_short_lateral_max_m
                inferred = cfg.link_allow_inferred and gap <= cfg.link_infer_gap_m and delta_z <= cfg.link_infer_dz_max_m and cos_first >= cfg.link_infer_min_cos and cos_second >= cfg.link_infer_min_cos and lateral <= cfg.link_infer_lateral_max_m
                if not (regular or short or inferred):
                    continue
                preliminary = gap / max(cfg.link_infer_gap_m, 1e-6) + 0.35 * (1.0 - min(cos_first, cos_second)) + 0.35 * lateral / max(cfg.link_short_lateral_max_m, 1e-6) + 0.35 * delta_z / max(cfg.link_infer_dz_max_m, 1e-6)
                if regular:
                    preliminary -= 0.20
                if short:
                    preliminary -= 0.10
                candidates.append({"first": first_index, "second": second_index, "gap": gap, "dz": delta_z, "cos_first": cos_first, "cos_second": cos_second, "lateral": lateral, "regular": regular, "short": short, "inferred": inferred, "pre_score": preliminary})
        if not candidates:
            break
        keep_indices = set()
        by_endpoint = {}
        for candidate_index, candidate in enumerate(candidates):
            for endpoint_index in (candidate["first"], candidate["second"]):
                by_endpoint.setdefault(endpoint_index, []).append(candidate_index)
        per_endpoint = max(1, int(cfg.link_max_candidates_per_endpoint))
        for candidate_indices in by_endpoint.values():
            candidate_indices.sort(key=lambda item: candidates[item]["pre_score"])
            keep_indices.update(candidate_indices[:per_endpoint])
        candidates = [candidates[index] for index in sorted(keep_indices)]

        scored = []
        for candidate in candidates:
            first = endpoints[candidate["first"]]
            second = endpoints[candidate["second"]]
            line_type = lines[first["line"]]["type"]
            if distance_by_type[line_type] is None:
                continue
            gap = candidate["gap"]
            pad = int(0.5 * gap / res) + 4
            row0 = max(0, min(first["rc"][0], second["rc"][0]) - pad)
            row1 = min(ny - 1, max(first["rc"][0], second["rc"][0]) + pad)
            col0 = max(0, min(first["rc"][1], second["rc"][1]) - pad)
            col1 = min(nx - 1, max(first["rc"][1], second["rc"][1]) + pad)
            sub_z = fill[row0:row1 + 1, col0:col1 + 1]
            sub_line = distance_by_type[line_type][row0:row1 + 1, col0:col1 + 1]
            sub_edge = edge_distance[line_type][row0:row1 + 1, col0:col1 + 1]
            height, width = sub_z.shape
            yy, xx = np.mgrid[0:height, 0:width]
            pixel_x, pixel_y = rc_to_xy(yy + row0, xx + col0)
            start_x, start_y = first["pos"]
            end_x, end_y = second["pos"]
            vector_x, vector_y = end_x - start_x, end_y - start_y
            chord2 = max(vector_x * vector_x + vector_y * vector_y, 1e-9)
            fraction = np.clip(((pixel_x - start_x) * vector_x + (pixel_y - start_y) * vector_y) / chord2, 0, 1)
            z_linear = first["z"] + fraction * (second["z"] - first["z"])
            cost = 1.0 + cfg.link_w_dist * sub_line + cfg.link_w_z * np.abs(sub_z - z_linear) + cfg.link_w_edge * sub_edge
            cost[~np.isfinite(sub_z)] = 25.0
            start_rc = (first["rc"][0] - row0, first["rc"][1] - col0)
            end_rc = (second["rc"][0] - row0, second["rc"][1] - col0)
            try:
                path, path_cost = route_through_array(cost, start_rc, end_rc, fully_connected=True, geometric=True)
            except Exception:
                continue
            path_edge = np.asarray([sub_edge[row, col] for row, col in path], dtype=float)
            edge_fraction = float(np.mean(path_edge <= cfg.link_edge_dist_m))
            support_ok = edge_fraction >= cfg.link_edge_min_fraction
            if candidate["regular"] and support_ok:
                mode, confidence = "regular_edge", 0.95
            elif candidate["short"] and support_ok:
                mode, confidence = "edge_supported_short", 0.88
            elif candidate["inferred"] and support_ok:
                mode, confidence = "edge_supported_long", 0.82
            elif candidate["regular"]:
                mode, confidence = "regular_low_evidence", 0.68
            elif candidate["inferred"] and cfg.link_allow_inferred:
                mode, confidence = "geometry_inferred", 0.45
            else:
                continue
            confidence += 0.10 * edge_fraction
            confidence -= 0.12 * min(1.0, gap / max(cfg.link_infer_gap_m, 1e-6))
            confidence = float(np.clip(confidence, 0.20, 0.99))
            score = path_cost / max(len(path), 1) + 0.015 * gap + 0.35 * (1.0 - min(candidate["cos_first"], candidate["cos_second"])) - 0.45 * edge_fraction + (0.35 if mode == "geometry_inferred" else 0.0)
            scored.append({"score": score, "mode": mode, "confidence": confidence, "first": first, "second": second, "gap": gap, "path": path, "row0": row0, "col0": col0, "edge_fraction": edge_fraction})
        if not scored:
            break
        scored.sort(key=lambda item: (_mode_rank(item["mode"]), item["score"]))
        used_endpoints = set()
        used_lines = set()
        merges = []
        for item in scored:
            first_id = (item["first"]["line"], item["first"]["which"])
            second_id = (item["second"]["line"], item["second"]["which"])
            if first_id in used_endpoints or second_id in used_endpoints:
                continue
            if item["first"]["line"] in used_lines or item["second"]["line"] in used_lines:
                continue
            merges.append(item)
            used_endpoints.add(first_id)
            used_endpoints.add(second_id)
            used_lines.add(item["first"]["line"])
            used_lines.add(item["second"]["line"])
        if not merges:
            break

        dropped = set()
        new_lines = []
        for item in merges:
            first_line = lines[item["first"]["line"]]
            second_line = lines[item["second"]["line"]]
            first_points = first_line["pts"]
            if item["first"]["which"] == "s":
                first_points = first_points[::-1]
            second_points = second_line["pts"]
            if item["second"]["which"] == "e":
                second_points = second_points[::-1]
            path_xy = np.asarray([rc_to_xy(row + item["row0"], col + item["col0"]) for row, col in item["path"]], dtype=float)
            if len(path_xy) > 2:
                path_xy = _simplify_path(path_xy[1:-1], max(0.8, cfg.simplify_tol_m * 0.8))
            else:
                path_xy = path_xy[:0]
            merged_points = np.vstack([first_points, path_xy, second_points]) if len(path_xy) else np.vstack([first_points, second_points])
            length = float(np.sum(np.hypot(np.diff(merged_points[:, 0]), np.diff(merged_points[:, 1]))))
            z_values = np.asarray([first_line.get("z", np.nan), second_line.get("z", np.nan)], dtype=float)
            z_value = float(np.nanmean(z_values)) if np.isfinite(z_values).any() else np.nan
            previous_modes = [first_line.get("_link_mode"), second_line.get("_link_mode")]
            if "geometry_inferred" in previous_modes or item["mode"] == "geometry_inferred":
                final_mode = "geometry_inferred"
            elif "edge_supported_long" in previous_modes or item["mode"] == "edge_supported_long":
                final_mode = "edge_supported_long"
            elif "edge_supported_short" in previous_modes or item["mode"] == "edge_supported_short":
                final_mode = "edge_supported_short"
            elif "regular_edge" in previous_modes or item["mode"] == "regular_edge":
                final_mode = "regular_edge"
            else:
                final_mode = item["mode"]
            previous_confidence = min(float(first_line.get("_confidence", 1.0)), float(second_line.get("_confidence", 1.0)))
            parent_sources = {first_line.get("_source", "platform_boundary"), second_line.get("_source", "platform_boundary")}
            if parent_sources == {"edge_seed"}:
                source = "edge_seed"
            elif "edge_seed" in parent_sources:
                source = "edge_seed_linked"
            elif parent_sources == {"multiscale_seed"}:
                source = "multiscale_seed"
            elif "multiscale_seed" in parent_sources:
                source = "multiscale_linked"
            else:
                source = "platform_boundary"
            new_lines.append({"type": first_line["type"], "pts": merged_points, "length": length, "z": z_value, "_linked": True, "_link_mode": final_mode, "_confidence": min(previous_confidence, item["confidence"]), "_edge_fraction": item["edge_fraction"], "_source": source})
            dropped.add(item["first"]["line"])
            dropped.add(item["second"]["line"])
            total_merged += 1
        for line_index, line in enumerate(lines):
            if line_index not in dropped:
                new_lines.append(line)
        lines = new_lines
        if verbose:
            inferred = sum(1 for item in merges if item["mode"] == "geometry_inferred")
            print(f"[enh-link] pass{pass_index + 1}: merged={len(merges)} inferred={inferred} -> {len(lines)} lines")
    if verbose:
        print(f"[enh-link] total merged pairs: {total_merged}")
    return lines
