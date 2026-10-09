# -*- coding: utf-8 -*-
"""
slopeline_core.py -- geometric crest/toe extraction from colored point clouds.

Pipeline: point cloud -> DSM grid -> slope -> platform/face -> ray-based arc
detection -> chain -> merge -> relief validation -> GeoJSON (ENU / WGS84) + PNG.
Pure numpy/scipy/skimage. No training involved.
"""
from __future__ import annotations
import sys, os, json, math, zipfile, glob
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import (disk, binary_opening, binary_closing,
                                binary_erosion, remove_small_objects)
from skimage.graph import route_through_array

from slopeline_enhanced import _direct_step_edges, link_lines_enhanced
from slopeline_autogap import recover_auto_gap_bridges

sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# 1. point-cloud readers (PLY binary/ascii, LAS 1.2-1.4 basics, Terra zip)
# ----------------------------------------------------------------------------
_TMAP = {"float": "f4", "float32": "f4", "float64": "f8", "double": "f8",
         "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
         "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
         "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}

@dataclass
class _Item:
    kind: str          # 'ply_file' | 'las_file'
    path: str
    zentry: Optional[str] = None   # name inside zip when source is a zip

def _discover(input_path: str):
    items = []
    if os.path.isdir(input_path):
        for ext in ("*.ply", "*.las", "*.PLY", "*.LAS", "*.zip"):
            items += [os.path.join(r, f) for r, _, fs in os.walk(input_path)
                      for f in fs if f.lower().endswith(ext.lower())]
    else:
        items = [input_path]
    items = sorted(set(items))
    out = []
    for it in items:
        if it.lower().endswith(".zip"):
            with zipfile.ZipFile(it) as z:
                for n in z.namelist():
                    if n.lower().endswith(".ply"):
                        out.append(_Item("ply_file", it, n))
                    elif n.lower().endswith(".las"):
                        out.append(_Item("las_file", it, n))
        elif it.lower().endswith(".ply"):
            out.append(_Item("ply_file", it))
        elif it.lower().endswith(".las"):
            out.append(_Item("las_file", it))
    if not out:
        raise RuntimeError(f"no .ply/.las/.zip found in {input_path}")
    return out

def _open(item: _Item):
    if item.zentry is not None:
        z = zipfile.ZipFile(item.path)
        f = z.open(item.zentry)
        return z, f
    return None, open(item.path, "rb")

class _PlyReader:
    """Generic binary (little/big endian) or ascii PLY vertex reader."""
    def __init__(self, item: _Item):
        self.item = item
        _, f = _open(item)
        try:
            head = b""
            while b"end_header" not in head:
                b = f.read(65536)
                if not b:
                    break
                head += b
            if b"end_header" not in head:
                raise RuntimeError("bad ply header")
        finally:
            f.close()
        text = head[: head.index(b"end_header")].decode("ascii", "replace")
        lines = [l.strip() for l in text.splitlines()]
        self.fmt = next(l.split()[1] for l in lines if l.startswith("format"))
        self.nverts = 0
        props = []
        in_vertex = False
        for l in lines:
            if l.startswith("element"):
                _, name, cnt = l.split()
                in_vertex = (name == "vertex")
                if in_vertex:
                    self.nverts = int(cnt)
            elif l.startswith("property") and in_vertex:
                parts = l.split()
                props.append((parts[1], parts[2]))
        self.props = props
        self.hs = head.index(b"end_header") + len(b"end_header\n")
        # map fields
        names = [p[1] for p in props]
        def find(pref):
            for p in props:
                if p[1].lower() in pref:
                    return p
            return None
        self.fx = find(("x",))
        self.frgb = [find(("red", "diffuse_red", "r")),
                     find(("green", "diffuse_green", "g")),
                     find(("blue", "diffuse_blue", "b"))]
        if self.fx is None:
            raise RuntimeError("no x property in ply")
        self.fy = find(("y",)); self.fz = find(("z",))
        if self.fy is None or self.fz is None:
            raise RuntimeError("no xyz in ply")
        if self.fmt.startswith("ascii"):
            return
        endian = "<" if self.fmt.endswith("little_endian") else ">"
        off = 0
        dt_list = []
        for typ, nm in props:
            if typ in _TMAP:
                dt_list.append((nm, endian + _TMAP[typ]))
            else:  # list property: unsupported for vertex payload
                raise RuntimeError(f"unsupported ply property type {typ}")
        # numpy structured dtype with itemsize = sum
        self.rec = sum(np.dtype(endian + _TMAP[t]).itemsize for t, _ in props)
        self.dtype = np.dtype(dt_list)

    def iter_chunks(self, chunk=4_000_000):
        z, f = _open(self.item)
        try:
            f.read(self.hs)
            if self.fmt.startswith("ascii"):
                raise RuntimeError("ascii ply streaming unsupported")
            carry = b""
            while True:
                want = chunk * self.rec - len(carry)
                b = f.read(want)
                if not b:
                    break
                body = carry + b
                n = len(body) // self.rec
                if n == 0:
                    carry = body
                    continue
                yield self._decode(body[: n * self.rec])
                carry = body[n * self.rec:]
            if carry:
                n = len(carry) // self.rec
                if n:
                    yield self._decode(carry[: n * self.rec])
        finally:
            f.close()
            if z is not None:
                z.close()

    def _decode(self, buf):
        a = np.frombuffer(buf, dtype=self.dtype)
        xyz = np.stack([a[self.fx[1]].astype(np.float64),
                        a[self.fy[1]].astype(np.float64),
                        a[self.fz[1]].astype(np.float64)], axis=1)
        rgb = None
        if all(r is not None for r in self.frgb):
            rgb = np.stack([a[self.frgb[0][1]].astype(np.uint8),
                            a[self.frgb[1][1]].astype(np.uint8),
                            a[self.frgb[2][1]].astype(np.uint8)], axis=1)
        return xyz, rgb

class _LasReader:
    def __init__(self, item: _Item):
        self.item = item
        z, f = _open(item)
        try:
            h = f.read(375)
        finally:
            f.close()
            if z is not None:
                z.close()
        if h[:4] != b"LASF":
            raise RuntimeError("bad las signature")
        self.ver = (h[24], h[25])
        self.off = struct_u32(h, 96)
        self.fmt = h[104]
        self.rec_len = struct_u16(h, 105)
        self.count = struct_u32(h, 107)
        if self.count == 0 and self.ver >= (1, 4) and len(h) >= 247 + 8:
            self.count = struct_u64(h, 247)
        self.scale = struct_d3(h, 131)
        self.offxyz = struct_d3(h, 155)
        if self.fmt not in (0, 1, 2, 3):
            raise RuntimeError(f"las point format {self.fmt} unsupported (use 0-3)")
        self.has_rgb = self.fmt in (2, 3)
        self.rgb_off = 20 if self.fmt == 2 else 28
        if self.rec_len == 0:
            self.rec_len = {0: 20, 1: 28, 2: 26, 3: 34}[self.fmt]
        self.total = self.rec_len * self.count

    def iter_chunks(self, chunk=4_000_000):
        z, f = _open(self.item)
        try:
            f.read(self.off)
            carry = b""
            while True:
                want = chunk * self.rec_len - len(carry)
                b = f.read(want)
                if not b:
                    break
                body = carry + b
                n = len(body) // self.rec_len
                if n == 0:
                    carry = body
                    continue
                yield self._decode(body[: n * self.rec_len])
                carry = body[n * self.rec_len:]
            if carry:
                n = len(carry) // self.rec_len
                if n:
                    yield self._decode(carry[: n * self.rec_len])
        finally:
            f.close()
            if z is not None:
                z.close()

    def _decode(self, buf):
        n = len(buf) // self.rec_len
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(n, self.rec_len)
        xi = np.ascontiguousarray(raw[:, 0:12]).view("<i4").reshape(-1, 3).astype(np.float64)
        xyz = xi * self.scale + self.offxyz
        rgb = None
        if self.has_rgb:
            r = raw[:, self.rgb_off:self.rgb_off+6]
            rgb = np.ascontiguousarray(r).view("<u2").reshape(-1, 3)
            rgb = (rgb >> 8).astype(np.uint8)
        return xyz, rgb

def struct_u32(b, o): return int(np.frombuffer(b[o:o+4], "<u4")[0])
def struct_u16(b, o): return int(np.frombuffer(b[o:o+2], "<u2")[0])
def struct_u64(b, o): return int(np.frombuffer(b[o:o+8], "<u8")[0])
def struct_d3(b, o): return np.frombuffer(b[o:o+24], "<f8").copy()

def iter_cloud(input_path):
    for it in _discover(input_path):
        rd = _PlyReader(it) if it.kind == "ply_file" else _LasReader(it)
        for xyz, rgb in rd.iter_chunks():
            yield xyz, rgb

# ----------------------------------------------------------------------------
# 2. grid building (single pass over discovered chunks after extent pass)
# ----------------------------------------------------------------------------
@dataclass
class Grid:
    zmax: np.ndarray      # (ny,nx) float32, NaN where empty
    rgb: np.ndarray       # (ny,nx,3) float32
    cnt: np.ndarray       # uint32
    res: float
    xmin: float
    ymax: float

def build_grid(input_path, res, verbose=True):
    """Aggregate at res/2 then median-combine 2x2 to `res` (robust DSM)."""
    ir = res / 2.0   # internal aggregation resolution
    # pass 1: bounds
    x0 = y0 = z0 = np.inf
    x1 = y1 = z1 = -np.inf
    for xyz, _ in iter_cloud(input_path):
        if xyz.size == 0:
            continue
        x0 = min(x0, float(xyz[:, 0].min())); x1 = max(x1, float(xyz[:, 0].max()))
        y0 = min(y0, float(xyz[:, 1].min())); y1 = max(y1, float(xyz[:, 1].max()))
        z0 = min(z0, float(xyz[:, 2].min())); z1 = max(z1, float(xyz[:, 2].max()))
    if not np.isfinite(x0):
        raise RuntimeError("no valid points")
    pad = 2.0
    xmin, xmax = x0 - pad, x1 + pad
    ymin, ymax = y0 - pad, y1 + pad
    nxi = int(math.ceil((xmax - xmin) / ir))
    nyi = int(math.ceil((ymax - ymin) / ir))
    if verbose:
        print(f"[grid] extent E[{x0:.1f},{x1:.1f}] N[{y0:.1f},{y1:.1f}] Z[{z0:.1f},{z1:.1f}] "
              f"agg@{ir}m -> out@{res}m ({nxi//2}x{nyi//2})")
    N = nxi * nyi
    zmax = np.full(N, -1e18, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.uint32)
    rs = np.zeros(N, dtype=np.float64)
    gs = np.zeros(N, dtype=np.float64)
    bs = np.zeros(N, dtype=np.float64)
    tot = 0
    for xyz, rgb in iter_cloud(input_path):
        if xyz.size == 0:
            continue
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > -1e6)
        if not ok.any():
            continue
        x, y, z = x[ok], y[ok], z[ok]
        col = np.floor((x - xmin) / ir).astype(np.int64)
        row = np.floor((ymax - y) / ir).astype(np.int64)
        m = (col >= 0) & (col < nxi) & (row >= 0) & (row < nyi)
        col, row, z = col[m], row[m], z[m]
        if col.size == 0:
            continue
        idx = row * nxi + col
        np.maximum.at(zmax, idx, z)
        np.add(cnt, np.bincount(idx, minlength=N).astype(np.uint32), out=cnt)
        if rgb is not None:
            rgb = rgb[ok][m]
            np.add(rs, np.bincount(idx, weights=rgb[:, 0].astype(np.float64), minlength=N), out=rs)
            np.add(gs, np.bincount(idx, weights=rgb[:, 1].astype(np.float64), minlength=N), out=gs)
            np.add(bs, np.bincount(idx, weights=rgb[:, 2].astype(np.float64), minlength=N), out=bs)
        tot += col.size
    if verbose:
        print(f"[grid] points aggregated: {tot/1e6:.2f}M")
    zmax = zmax.reshape(nyi, nxi).astype(np.float32)
    cnt = cnt.reshape(nyi, nxi)
    nz = cnt > 0
    rgbm = np.stack([rs, gs, bs], -1).reshape(nyi, nxi, 3)
    rgbm = np.where(nz[..., None], rgbm / np.maximum(cnt[..., None], 1), 0).astype(np.float32)
    zmax[~nz] = np.nan
    # combine 2x2 blocks -> final res
    f = 2
    nyf, nxf = nyi // f, nxi // f
    zz = zmax[:nyf*f, :nxf*f].reshape(nyf, f, nxf, f)
    cc = cnt[:nyf*f, :nxf*f].reshape(nyf, f, nxf, f)
    rr = rgbm[:nyf*f, :nxf*f].reshape(nyf, f, nxf, f, 3)
    with np.errstate(all="ignore"):
        zout = np.nanmedian(zz, axis=(1, 3)).astype(np.float32)
    cout = cc.sum(axis=(1, 3))
    rout = rr.mean(axis=(1, 3)).astype(np.float32)
    nzout = cout > 0
    zout[~nzout] = np.nan
    rout[~nzout] = 0
    return Grid(zout, rout, cout.astype(np.uint32), float(res), float(xmin), float(ymax))

# ----------------------------------------------------------------------------
# 3. DSM prep
# ----------------------------------------------------------------------------
def prepare_dsm(g: Grid, smooth_sigma_cells=1.0, verbose=True):
    zmax = g.zmax
    valid = np.isfinite(zmax)
    dist, idx = ndi.distance_transform_edt(~valid, return_indices=True, return_distances=True)
    filled = zmax[tuple(idx)]
    data_region = dist <= 3.0
    dsm = ndi.median_filter(filled, size=3, mode="nearest")
    dsm = ndi.gaussian_filter(dsm, sigma=smooth_sigma_cells)
    dsm[~data_region] = np.nan
    gy, gx = np.gradient(dsm, g.res, g.res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~data_region] = np.nan
    if verbose:
        ok = data_region & np.isfinite(slope)
        print(f"[dsm] slope med={np.nanmedian(slope[ok]):.1f}deg  "
              f"p90={np.nanpercentile(slope[ok], 90):.1f}deg")
    return dsm, slope, data_region

# ----------------------------------------------------------------------------
# 4. line extraction
# ----------------------------------------------------------------------------
def extract_lines(dsm, slope, data_region, cfg, xmin, ymax, verbose=True):
    res = cfg.grid_res_m
    T1 = cfg.platform_slope_deg
    T2 = cfg.face_slope_deg
    DROP = cfg.min_step_drop_m
    RAY_M = cfg.ray_m
    ny, nx = dsm.shape

    plat0 = (slope < T1) & data_region
    plat = binary_closing(plat0, disk(2))
    plat = remove_small_objects(binary_opening(plat, disk(1)),
                                min_size=cfg.min_platform_px)
    L, nlab = ndi.label(plat, structure=np.ones((3, 3)))
    if verbose:
        print(f"[class] platform comps={nlab} px={int(plat.sum()):,}")

    # ray offsets in cells
    ray_cells = [max(1, int(round(m / res))) for m in RAY_M]
    str_ = np.ones((3, 3), bool)
    crest_px, toe_px = [], []
    slices = ndi.find_objects(L)
    for si, sl in enumerate(slices):
        if sl is None:
            continue
        comp = L[sl] == (si + 1)
        ny0, nx0 = comp.shape
        bdy = comp & ~binary_erosion(comp, str_)
        if bdy.sum() == 0:
            continue
        W = max(ray_cells) + 1
        cp = np.pad(comp, W, constant_values=False)
        dsm_sub = dsm[sl]
        vp = np.pad(np.isfinite(dsm_sub), W, constant_values=False)
        dp = np.pad(np.where(np.isfinite(dsm_sub), dsm_sub, np.nan), W,
                    constant_values=np.nan)
        crest = np.zeros_like(comp, bool)
        toe = np.zeros_like(comp, bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                oz = np.zeros((ny0, nx0)); ocnt = np.zeros((ny0, nx0))
                for k in ray_cells:
                    rr = W + dr * k; cc = W + dc * k
                    ncomp = cp[rr:rr+ny0, cc:cc+nx0]
                    nvalid = vp[rr:rr+ny0, cc:cc+nx0]
                    ndsm = dp[rr:rr+ny0, cc:cc+nx0]
                    outside = ~ncomp & nvalid
                    oz += np.where(outside, ndsm, 0.0)
                    ocnt += outside
                ok2 = ocnt >= max(1, len(ray_cells) // 2)
                with np.errstate(all="ignore"):
                    mean_out = oz / np.maximum(ocnt, 1)
                z_p = dsm_sub
                is_bdy = bdy & ok2
                toe |= is_bdy & (mean_out - z_p > DROP)
                crest |= is_bdy & (z_p - mean_out > DROP)
        yy, xx = np.where(crest)
        if len(yy):
            crest_px.append((yy + sl[0].start, xx + sl[1].start))
        yy, xx = np.where(toe)
        if len(yy):
            toe_px.append((yy + sl[0].start, xx + sl[1].start))
    if verbose:
        print(f"[arcs] crest px={sum(len(a[0]) for a in crest_px):,}  "
              f"toe px={sum(len(a[0]) for a in toe_px):,}")

    # chaining --------------------------------------------------------------
    def rc_to_xy(rows, cols):
        x = xmin + (cols + 0.5) * res
        y = ymax - (rows + 0.5) * res
        return x, y

    def dp_simplify(pts, tol):
        if len(pts) < 3:
            return pts
        keep = np.zeros(len(pts), bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            p1, p2 = pts[i], pts[j]
            seg = p2 - p1
            L2 = seg @ seg
            if L2 < 1e-12:
                d = np.hypot(pts[i+1:j, 0] - p1[0], pts[i+1:j, 1] - p1[1])
            else:
                t = np.clip(((pts[i+1:j] - p1) @ seg) / L2, 0, 1)
                proj = p1 + t[:, None] * seg
                d = np.hypot(pts[i+1:j, 0] - proj[:, 0], pts[i+1:j, 1] - proj[:, 1])
            k = int(np.argmax(d))
            if d[k] > tol:
                keep[i+1+k] = True
                stack.append((i, i+1+k))
                stack.append((i+1+k, j))
        return pts[keep]

    def chain_arc(rows, cols, min_arc_px=None, min_line_m=None):
        min_px = cfg.min_arc_px if min_arc_px is None else int(min_arc_px)
        min_length = cfg.min_line_m if min_line_m is None else float(min_line_m)
        m = len(rows)
        if m < min_px:
            return []
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        H, W = r1 - r0 + 5, c1 - c0 + 5
        idm = np.full((H, W), -1, dtype=np.int32)
        rr = rows - r0 + 2
        cc = cols - c0 + 2
        idm[rr, cc] = np.arange(m, dtype=np.int32)
        pad = np.pad(idm, 2, constant_values=-1)
        nbr_list = []
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                if dr == 0 and dc == 0:
                    continue
                nbr_list.append(pad[rr+2+dr, cc+2+dc])
        nbs = np.stack(nbr_list, axis=1)
        adj = []
        deg = np.zeros(m, np.int8)
        for i in range(m):
            u = {int(x) for x in nbs[i] if x >= 0}
            adj.append(u)
            deg[i] = len(u)
        used = np.zeros(m, bool)
        chains = []
        order = list(np.where(deg <= 1)[0]) + list(np.where(deg > 1)[0])
        for s in order:
            if used[s]:
                continue
            ch = [s]
            used[s] = True
            prev = -1
            cur = s
            guard = 0
            while guard < m:
                guard += 1
                nxt = [j for j in adj[cur] if j != prev and not used[j]]
                if not nxt:
                    break
                j = nxt[0]
                ch.append(j)
                used[j] = True
                prev, cur = cur, j
            if len(ch) >= min_px:
                chains.append(ch)
        out = []
        for ch in chains:
            rr2 = rows[ch]; cc2 = cols[ch]
            x, y = rc_to_xy(rr2, cc2)
            pts = np.stack([x, y], axis=1)
            pts = dp_simplify(pts, cfg.simplify_tol_m)
            if len(pts) < 2:
                continue
            length = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
            if length < min_length:
                continue
            zmed = float(np.nanmedian(dsm[rr2, cc2]))
            out.append({"pts": pts, "length": length, "z": zmed})
        return out

    lines = []
    for comp_idx, (cr, to) in enumerate(zip(crest_px, toe_px)):
        for typ, px in (("crest", cr), ("toe", to)):
            if len(px[0]) == 0:
                continue
            for r in chain_arc(px[0], px[1]):
                r["type"] = typ
                r["_source"] = "platform_boundary"
                lines.append(r)
    if cfg.edge_seed_enabled and cfg.edge_seed_regions:
        region_params = list(getattr(cfg, "edge_seed_region_params", []) or [])
        for region_index, region in enumerate(cfg.edge_seed_regions):
            if region is None or len(region) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in region]
            col0 = int(np.clip(np.floor((min(x0, x1) - xmin) / res), 0, nx - 1))
            col1 = int(np.clip(np.ceil((max(x0, x1) - xmin) / res), 0, nx))
            row0 = int(np.clip(np.floor((ymax - max(y0, y1)) / res), 0, ny - 1))
            row1 = int(np.clip(np.ceil((ymax - min(y0, y1)) / res), 0, ny))
            local_mask = np.zeros_like(data_region, dtype=bool)
            local_mask[row0:row1, col0:col1] = True
            if not local_mask.any():
                continue

            override = region_params[region_index] if region_index < len(region_params) else None
            if not isinstance(override, dict):
                override = {}
            center_slope_deg = float(override.get(
                "center_slope_deg", cfg.edge_seed_center_slope_deg))
            drop_m = float(override.get("drop_m", cfg.edge_seed_drop_m))
            min_line_m = float(override.get(
                "min_line_m", cfg.edge_seed_min_line_m))
            min_px = max(1, int(override.get("min_px", cfg.edge_seed_min_px)))
            dedup_dist_m = float(override.get(
                "dedup_dist_m", cfg.edge_seed_dedup_dist_m))

            edge_crest, edge_toe = _direct_step_edges(
                dsm,
                res,
                cfg.ray_m,
                drop_m,
                center_slope_deg,
            )
            for typ, edge_mask in (("crest", edge_crest), ("toe", edge_toe)):
                seed_mask = edge_mask & data_region & local_mask
                existing_mask = np.zeros_like(seed_mask)
                for line in lines:
                    if line["type"] != typ:
                        continue
                    rows = np.clip(((ymax - line["pts"][:, 1]) / res).astype(int), 0, ny - 1)
                    cols = np.clip(((line["pts"][:, 0] - xmin) / res).astype(int), 0, nx - 1)
                    existing_mask[rows, cols] = True
                if dedup_dist_m > 0 and existing_mask.any():
                    distance_to_existing = ndi.distance_transform_edt(~existing_mask)
                    seed_mask &= distance_to_existing > dedup_dist_m
                seed_mask = remove_small_objects(seed_mask, min_size=min_px)
                seed_rows, seed_cols = np.where(seed_mask)
                for line in chain_arc(
                    seed_rows,
                    seed_cols,
                    min_arc_px=min_px,
                    min_line_m=min_line_m,
                ):
                    line["type"] = typ
                    line["_source"] = "edge_seed"
                    line["_seed_region_index"] = region_index
                    lines.append(line)

    if verbose:
        nc = sum(1 for l in lines if l["type"] == "crest")
        nt = sum(1 for l in lines if l["type"] == "toe")
        print(f"[lines] raw crest={nc} toe={nt} tot_len={sum(l['length'] for l in lines):.0f}m")
    return lines


def _sample_line_points(line: dict, step_m: float = 1.0) -> np.ndarray:
    """Resample one polyline at a regular spacing for evidence scoring."""
    points = np.asarray(line["pts"], dtype=float)
    if len(points) < 2:
        return points
    segment = np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))
    if segment[-1] <= 1e-6:
        return points
    distances = np.arange(0.0, segment[-1] + step_m, step_m)
    distances = distances[distances <= segment[-1]]
    if len(distances) == 0 or distances[-1] < segment[-1]:
        distances = np.append(distances, segment[-1])
    return np.column_stack([
        np.interp(distances, np.concatenate([[0.0], segment]), points[:, 0]),
        np.interp(distances, np.concatenate([[0.0], segment]), points[:, 1]),
    ])


def _line_mask(lines, line_type: str, shape, xmin: float, ymax: float,
               res: float) -> np.ndarray:
    """Rasterize same-type line centerlines for spatial de-duplication."""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    for line in lines:
        if line["type"] != line_type:
            continue
        points = _sample_line_points(line, step_m=res)
        rows = np.clip(((ymax - points[:, 1]) / res).astype(int), 0, ny - 1)
        cols = np.clip(((points[:, 0] - xmin) / res).astype(int), 0, nx - 1)
        mask[rows, cols] = True
    return mask


def augment_multiscale_lines(lines, dsm, data_region, cfg, xmin, ymax,
                             verbose=True):
    """Recover missed benches with relaxed platform seeds.

    A relaxed candidate is accepted only when it follows a direct point-cloud
    step edge and is not a spatial duplicate of a strict line. Candidates near
    the existing network may bridge gaps; long isolated candidates need strong
    edge support.
    """
    if not cfg.multiscale_enabled:
        return lines
    relaxed_cfg = replace(
        cfg,
        platform_slope_deg=cfg.multiscale_slope_deg,
        min_platform_px=cfg.multiscale_min_platform_px,
        min_line_m=cfg.multiscale_min_line_m,
        edge_seed_enabled=False,
    )
    gy, gx = np.gradient(dsm, cfg.grid_res_m, cfg.grid_res_m)
    relaxed_slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relaxed_slope[~data_region] = np.nan
    relaxed = extract_lines(
        dsm,
        relaxed_slope,
        data_region,
        relaxed_cfg,
        xmin,
        ymax,
        verbose=False,
    )
    if not relaxed:
        return lines

    res = cfg.grid_res_m
    ny, nx = dsm.shape
    edge_crest, edge_toe = _direct_step_edges(
        dsm,
        res,
        cfg.ray_m,
        cfg.multiscale_drop_m,
        cfg.multiscale_edge_center_slope_deg,
    )
    edge_masks = {"crest": edge_crest, "toe": edge_toe}
    evidence_maps = {
        line_type: ndi.distance_transform_edt(~mask) * res
        for line_type, mask in edge_masks.items()
    }

    spatial = {}
    for line_type in ("crest", "toe"):
        mask = _line_mask(lines, line_type, dsm.shape, xmin, ymax, res)
        spatial[line_type] = (
            mask,
            ndi.distance_transform_edt(~mask) * res,
        )

    accepted = []
    candidates = sorted(relaxed, key=lambda item: item["length"], reverse=True)
    for candidate in candidates:
        line_type = candidate["type"]
        points = _sample_line_points(candidate, step_m=1.0)
        if len(points) == 0:
            continue
        rows = np.clip(((ymax - points[:, 1]) / res).astype(int), 0, ny - 1)
        cols = np.clip(((points[:, 0] - xmin) / res).astype(int), 0, nx - 1)

        edge_distance = evidence_maps[line_type][rows, cols]
        edge_fraction = float(np.mean(
            edge_distance <= cfg.multiscale_edge_dist_m))
        if edge_fraction < cfg.multiscale_edge_min_fraction:
            continue

        existing_distance = spatial[line_type][1][rows, cols]
        if float(np.median(existing_distance)) <= cfg.multiscale_dedup_dist_m:
            continue

        endpoint_gap = min(
            float(spatial[line_type][1][rows[0], cols[0]]),
            float(spatial[line_type][1][rows[-1], cols[-1]]),
        )
        network_supported = endpoint_gap <= cfg.multiscale_network_gap_m
        stand_alone_supported = (
            candidate["length"] >= cfg.multiscale_isolated_min_length_m
            and edge_fraction >= cfg.multiscale_isolated_edge_fraction
        )
        if not (network_supported or stand_alone_supported):
            continue

        candidate["_source"] = "multiscale_seed"
        candidate["_confidence"] = float(np.clip(
            0.62 + 0.30 * edge_fraction, 0.20, 0.95))
        candidate["_edge_fraction"] = edge_fraction
        accepted.append(candidate)

        extra_mask = _line_mask(accepted, line_type, dsm.shape,
                                xmin, ymax, res)
        combined_mask = spatial[line_type][0] | extra_mask
        spatial[line_type] = (
            combined_mask,
            ndi.distance_transform_edt(~combined_mask) * res,
        )

    if verbose:
        added_length = sum(item["length"] for item in accepted)
        print(f"[multiscale] relaxed={len(relaxed)} accepted={len(accepted)} "
              f"added_len={added_length:.0f}m")
    return lines + accepted

# ----------------------------------------------------------------------------
# 5. relief validation (keep only lines next to a real >=min_face_height step)
# ----------------------------------------------------------------------------
def validate_lines(lines, dsm, g, min_face_height_m, verbose=True):
    ny, nx = dsm.shape
    res = g.res

    def dsm_at(x, y):
        c = int(np.clip((x - g.xmin) / res, 0, nx - 1))
        r = int(np.clip((g.ymax - y) / res, 0, ny - 1))
        return dsm[r, c]

    keep = []
    for l in lines:
        pts = l["pts"]
        seg = np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
        if len(seg) == 0:
            continue
        dists = np.arange(0, seg[-1], 2.0)
        xs = np.interp(dists, np.concatenate([[0], seg]), pts[:, 0])
        ys = np.interp(dists, np.concatenate([[0], seg]), pts[:, 1])
        rels = []
        for x, y in zip(xs, ys):
            z0 = dsm_at(x, y)
            if not np.isfinite(z0):
                continue
            j = int(np.argmin((pts[:, 0]-x)**2 + (pts[:, 1]-y)**2))
            j0, j1 = max(0, j-2), min(len(pts)-1, j+2)
            dx = pts[j1, 0] - pts[j0, 0]; dy = pts[j1, 1] - pts[j0, 1]
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            nx_, ny_ = -dy / L, dx / L
            vm, vp = [], []
            for k in (2, 3, 4, 5, 6, 7, 8):
                vm.append(dsm_at(x + nx_*k, y + ny_*k))
                vp.append(dsm_at(x - nx_*k, y - ny_*k))
            vm = [v for v in vm if np.isfinite(v)]
            vp = [v for v in vp if np.isfinite(v)]
            if not vm or not vp:
                continue
            high = max(np.max(vm), np.max(vp))
            low = min(np.min(vm), np.min(vp))
            rels.append(z0 - low if l["type"] == "crest" else high - z0)
        if rels and float(np.median(rels)) >= min_face_height_m:
            keep.append(l)
    if verbose:
        nc = sum(1 for l in keep if l["type"] == "crest")
        nt = sum(1 for l in keep if l["type"] == "toe")
        print(f"[valid] crest={nc} toe={nt} (relief>={min_face_height_m}m)")
    return keep


# ----------------------------------------------------------------------------
# 5b. fragment linking: connect endpoint pairs along a break-line corridor
# ----------------------------------------------------------------------------
def link_lines(lines, dsm, g, cfg, verbose=True):
    """Greedy 1-to-1 endpoint matching; gaps are bridged by a lowest-cost path
    that hugs existing break-line pixels instead of a straight line."""
    if cfg.link_gap_m <= 0 or len(lines) < 2:
        return lines
    res = g.res
    ny, nx = dsm.shape
    # fill DSM holes for path cost
    valid = np.isfinite(dsm)
    if not valid.any():
        return lines
    dist_h, idx_h = ndi.distance_transform_edt(~valid, return_indices=True,
                                               return_distances=True)
    fill = dsm[tuple(idx_h)]
    fill[dist_h > 6] = np.nan

    def xy_to_rc(x, y):
        c = int(np.clip((x - g.xmin) / res, 0, nx - 1))
        r = int(np.clip((g.ymax - y) / res, 0, ny - 1))
        return r, c

    def rc_to_xy(r, c):
        return g.xmin + (c + 0.5) * res, g.ymax - (r + 0.5) * res

    n_link = 0
    for _pass in range(cfg.link_passes):
        # raster of same-type fragments -> distance transform
        dist_by_type = {}
        for typ in ("crest", "toe"):
            mask = np.zeros((ny, nx), bool)
            for l in lines:
                if l["type"] != typ:
                    continue
                rr = np.clip(((g.ymax - l["pts"][:, 1]) / res).astype(int), 0, ny - 1)
                cc = np.clip(((l["pts"][:, 0] - g.xmin) / res).astype(int), 0, nx - 1)
                mask[rr, cc] = True
            dist_by_type[typ] = ndi.distance_transform_edt(~mask) if mask.any() else None

        # endpoints with robust outward tangents
        ends = []
        for li, l in enumerate(lines):
            p = l["pts"]
            seg = np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))
            for which in ("s", "e"):
                if which == "s":
                    pos = p[0]
                    k = int(np.searchsorted(seg, min(15.0, max(seg[-1] * 0.4, 0.1)))) if len(seg) else 0
                    k = min(k + 1, len(p) - 1)
                    tang = p[0] - p[k]
                else:
                    pos = p[-1]
                    k = int(np.searchsorted(seg, max(seg[-1] - min(15.0, max(seg[-1] * 0.4, 0.1)), 0.0))) if len(seg) else len(p) - 1
                    k = max(k - 1, 0)
                    tang = p[-1] - p[k]
                n = np.linalg.norm(tang)
                if n < 1e-6:
                    continue
                r, c = xy_to_rc(pos[0], pos[1])
                z = fill[r, c]
                if not np.isfinite(z):
                    continue
                ends.append({"line": li, "which": which, "pos": pos,
                             "tan": tang / n, "z": float(z), "rc": (r, c)})

        # candidate pairs
        cands = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                ea, eb = ends[a], ends[b]
                if ea["line"] == eb["line"]:
                    continue
                if lines[ea["line"]]["type"] != lines[eb["line"]]["type"]:
                    continue
                dvec = eb["pos"] - ea["pos"]
                gap = float(np.hypot(*dvec))
                if gap < 0.5 or gap > cfg.link_gap_m:
                    continue
                if abs(ea["z"] - eb["z"]) > cfg.link_dz_max_m:
                    continue
                u = dvec / gap
                if float(ea["tan"] @ u) < cfg.link_min_cos:
                    continue
                if float(eb["tan"] @ (-u)) < cfg.link_min_cos:
                    continue
                lat = abs(float(np.cross(u, ea["tan"]))) * gap
                if lat > cfg.link_lateral_max_m:
                    continue
                cands.append((gap, a, b))
        if not cands:
            break
        # evaluate cost paths (local windows), keep best per endpoint pair
        scored = []
        distmap = dist_by_type
        for gap, a, b in cands:
            ea, eb = ends[a], ends[b]
            if distmap[lines[ea["line"]]["type"]] is None:
                continue
            r0 = max(0, min(ea["rc"][0], eb["rc"][0]) - int(0.5 * gap / res) - 3)
            r1 = min(ny - 1, max(ea["rc"][0], eb["rc"][0]) + int(0.5 * gap / res) + 3)
            c0 = max(0, min(ea["rc"][1], eb["rc"][1]) - int(0.5 * gap / res) - 3)
            c1 = min(nx - 1, max(ea["rc"][1], eb["rc"][1]) + int(0.5 * gap / res) + 3)
            sub_d = distmap[lines[ea["line"]]["type"]][r0:r1+1, c0:c1+1]
            sub_z = fill[r0:r1+1, c0:c1+1]
            hh, ww = sub_z.shape
            yy, xx = np.mgrid[0:hh, 0:ww]
            px, py = rc_to_xy(yy + r0, xx + c0)
            ax, ay = ea["pos"]; bx, by = eb["pos"]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / max(L2, 1e-9), 0, 1)
            z_lin = ea["z"] + t * (eb["z"] - ea["z"])
            cost = 1.0 + cfg.link_w_dist * sub_d + cfg.link_w_z * np.abs(sub_z - z_lin)
            cost[~np.isfinite(sub_z)] = 25.0
            s_rc = (ea["rc"][0] - r0, ea["rc"][1] - c0)
            e_rc = (eb["rc"][0] - r0, eb["rc"][1] - c0)
            try:
                path, cst = route_through_array(cost, s_rc, e_rc,
                                                fully_connected=True, geometric=True)
            except Exception:
                continue
            ncell = max(len(path), 1)
            scored.append((cst / ncell + 0.02 * gap, a, b, path, r0, c0, cst))
        if not scored:
            break
        scored.sort(key=lambda t: t[0])
        used_end = set()
        used_line = set()
        merges = []
        for score, a, b, path, r0, c0, cst in scored:
            if a in used_end or b in used_end:
                continue
            ea, eb = ends[a], ends[b]
            if ea["line"] in used_line or eb["line"] in used_line:
                continue
            merges.append((ea["line"], ea["which"], eb["line"], eb["which"], path, r0, c0))
            used_end.add(a); used_end.add(b)
            used_line.add(ea["line"]); used_line.add(eb["line"])
        if not merges:
            break
        # apply merges (rebuild list, skip consumed lines)
        consumed = {}
        new_lines = []
        for li, l in enumerate(lines):
            consumed.setdefault(li, li)
        drop = set()
        group = {}
        for k, (li, wi, lj, wj, path, r0, c0) in enumerate(merges):
            if li in drop or lj in drop:
                continue
            A = lines[li]["pts"]
            A = A[::-1] if wi == "s" else A
            B = lines[lj]["pts"]
            B = B[::-1] if wj == "e" else B
            path_xy = np.array([rc_to_xy(r + r0, c + c0) for r, c in path])
            if len(path_xy) > 2:
                path_xy = path_xy[1:-1]
            else:
                path_xy = path_xy[:0]
            C = np.vstack([A, path_xy, B]) if len(path_xy) else np.vstack([A, B])
            la = float(np.sum(np.hypot(np.diff(A[:, 0]), np.diff(A[:, 1]))))
            lb = float(np.sum(np.hypot(np.diff(B[:, 0]), np.diff(B[:, 1]))))
            lt = sum(abs(lines[li].get("_z", 0)) for _ in [0])  # noop
            za = lines[li].get("z", np.nan); zb = lines[lj].get("z", np.nan)
            z = np.nanmean([za, zb])
            length = float(np.sum(np.hypot(np.diff(C[:, 0]), np.diff(C[:, 1]))))
            new_lines.append({"type": lines[li]["type"], "pts": C, "length": length,
                              "z": float(z), "_linked": True})
            drop.add(li); drop.add(lj)
            n_link += 1
        for li, l in enumerate(lines):
            if li not in drop:
                new_lines.append(l)
        lines = new_lines
        if verbose:
            print(f"[link] pass{_pass+1}: merged {len(merges)} pairs -> {len(lines)} lines")
    if verbose:
        print(f"[link] total merged pairs: {n_link}")
    return lines

def link_lines_local(lines, dsm, g, cfg, verbose=True):
    """Relaxed linking restricted to user-specified recall-boost regions."""
    if not cfg.edge_seed_local_link_enabled or not cfg.edge_seed_regions or len(lines) < 2:
        return lines
    buffer_m = max(0.0, float(cfg.edge_seed_local_link_buffer_m))
    selected_indices = []
    for line_index, line in enumerate(lines):
        points = line["pts"]
        selected = False
        for region in cfg.edge_seed_regions:
            if region is None or len(region) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in region]
            xmin = min(x0, x1) - buffer_m
            xmax = max(x0, x1) + buffer_m
            ymin = min(y0, y1) - buffer_m
            ymax = max(y0, y1) + buffer_m
            if np.any((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                      (points[:, 1] >= ymin) & (points[:, 1] <= ymax)):
                selected = True
                break
        if selected:
            selected_indices.append(line_index)
    if len(selected_indices) < 2:
        return lines
    selected_set = set(selected_indices)
    selected_lines = [lines[index] for index in selected_indices]
    local_cfg = replace(
        cfg,
        link_gap_m=cfg.edge_seed_local_link_gap_m,
        link_dz_max_m=cfg.edge_seed_local_link_dz_max_m,
        link_min_cos=cfg.edge_seed_local_link_min_cos,
        link_lateral_max_m=cfg.edge_seed_local_link_lateral_max_m,
        link_short_gap_m=max(cfg.link_short_gap_m, cfg.edge_seed_local_link_gap_m),
        link_short_dz_max_m=max(cfg.link_short_dz_max_m, cfg.edge_seed_local_link_dz_max_m),
        link_short_lateral_max_m=max(cfg.link_short_lateral_max_m, cfg.edge_seed_local_link_lateral_max_m),
        link_infer_dz_max_m=max(cfg.link_infer_dz_max_m, cfg.edge_seed_local_link_dz_max_m),
        link_infer_min_cos=min(cfg.link_infer_min_cos, -0.10),
        link_infer_lateral_max_m=max(cfg.link_infer_lateral_max_m, cfg.edge_seed_local_link_lateral_max_m),
    )
    linked_lines = link_lines_enhanced(selected_lines, dsm, g, local_cfg, verbose=verbose)
    output = [line for index, line in enumerate(lines) if index not in selected_set]
    output.extend(linked_lines)
    return output

# ----------------------------------------------------------------------------
# 6. export
# ----------------------------------------------------------------------------
def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))))


def _deduplicate_polyline(points: np.ndarray, tolerance_m: float = 0.05) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [0]
    for index in range(1, len(points)):
        if np.hypot(*(points[index] - points[keep[-1]])) > tolerance_m:
            keep.append(index)
    if keep[-1] != len(points) - 1:
        keep.append(len(points) - 1)
    return points[keep]


def _closest_point_on_polyline(points: np.ndarray, query: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the shortest distance and closest point on a polyline."""
    if len(points) == 1:
        return float(np.hypot(*(points[0] - query))), points[0]
    starts = points[:-1]
    ends = points[1:]
    segments = ends - starts
    lengths2 = np.sum(segments * segments, axis=1)
    fractions = np.divide(
        np.sum((query - starts) * segments, axis=1),
        np.maximum(lengths2, 1e-12),
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * segments
    distances = np.hypot(projections[:, 0] - query[0], projections[:, 1] - query[1])
    index = int(np.argmin(distances))
    return float(distances[index]), projections[index]


def apply_guided_links(lines: list[dict], cfg: Config, verbose: bool = True) -> list[dict]:
    """Add user-guided ENU connectors without merging unrelated source lines."""
    constraints = list(getattr(cfg, "guided_links", []) or [])
    if not constraints:
        return lines
    current = list(lines)
    applied = 0
    for constraint_index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            continue
        guide = np.asarray(constraint.get("guide", []), dtype=float)
        if guide.ndim != 2 or guide.shape[0] < 2:
            continue
        line_type = str(constraint.get("type", "crest"))
        max_start = float(constraint.get("max_endpoint_dist_start_m",
                         constraint.get("max_endpoint_dist_m", 50.0)))
        max_end = float(constraint.get("max_endpoint_dist_end_m",
                       constraint.get("max_endpoint_dist_m", 50.0)))
        max_gap = float(constraint.get("max_gap_m", constraint.get("max_anchor_gap_m", 100.0)))
        max_dz = float(constraint.get("max_dz_m", np.inf))
        guide_length = _polyline_length(guide)

        start_anchors = []
        end_anchors = []
        for line_index, line in enumerate(current):
            if line.get("type") != line_type or line.get("_source") == "manual_constraint":
                continue
            distance_start, point_start = _closest_point_on_polyline(line["pts"], guide[0])
            if distance_start <= max_start:
                start_anchors.append({
                    "line": line_index, "distance": distance_start, "point": point_start,
                    "z": float(line.get("z", np.nan)),
                })
            distance_end, point_end = _closest_point_on_polyline(line["pts"], guide[-1])
            if distance_end <= max_end:
                end_anchors.append({
                    "line": line_index, "distance": distance_end, "point": point_end,
                    "z": float(line.get("z", np.nan)),
                })

        best = None
        for first in start_anchors:
            for second in end_anchors:
                if first["line"] == second["line"]:
                    continue
                anchor_gap = float(np.hypot(*(first["point"] - second["point"])))
                if anchor_gap < 1.0 or anchor_gap > max_gap:
                    continue
                anchor_dz = abs(first.get("z", np.nan) - second.get("z", np.nan))
                if np.isfinite(anchor_dz) and anchor_dz > max_dz:
                    continue
                score = first["distance"] + second["distance"] + 0.03 * anchor_gap
                if best is None or score < best["score"]:
                    best = {
                        "first": first, "second": second, "score": score,
                        "anchor_gap": anchor_gap, "anchor_dz": anchor_dz,
                    }
        if best is None:
            if verbose:
                print(f"[guided] {constraint.get('id', constraint_index)}: skipped, no eligible line pair")
            continue

        first = best["first"]
        second = best["second"]
        connector = _deduplicate_polyline(np.vstack([
            first["point"][None, :], guide, second["point"][None, :]
        ]))
        z_values = [first.get("z", np.nan), second.get("z", np.nan)]
        z_value = float(np.nanmean(z_values)) if np.isfinite(z_values).any() else float(np.nanmedian(guide[:, 1]))
        new_line = {
            "type": line_type,
            "pts": connector,
            "length": _polyline_length(connector),
            "z": z_value,
            "_source": "manual_constraint",
            "_link_mode": "manual_guided",
            "_confidence": 0.90,
            "_constraint_id": constraint.get("id", str(constraint_index)),
            "_anchor_line_start": first["line"],
            "_anchor_line_end": second["line"],
            "_anchor_dist_start_m": first["distance"],
            "_anchor_dist_end_m": second["distance"],
            "_anchor_gap_m": best["anchor_gap"],
            "_anchor_dz_m": best["anchor_dz"],
            "_guide_length_m": guide_length,
        }
        current.append(new_line)
        applied += 1
        if verbose:
            print(f"[guided] {constraint.get('id', constraint_index)}: connector "
                  f"{first['point'].round(1)} -> {second['point'].round(1)}, "
                  f"anchor_dist={first['distance']:.1f}/{second['distance']:.1f}m, "
                  f"anchor_gap={best['anchor_gap']:.1f}m, "
                  f"dz={best['anchor_dz']:.1f}m")
    if verbose:
        print(f"[guided] constraints={len(constraints)} applied={applied}")
    return current


def export_geojson(lines, out_path, crs, origin):
    feats = []
    for l in lines:
        link_mode = l.get("_link_mode", "observed")
        confidence = float(l.get("_confidence", 1.0))
        if link_mode == "geometry_inferred":
            connection_mode = "inferred"
        elif link_mode == "observed":
            connection_mode = "observed"
        elif link_mode == "manual_guided":
            connection_mode = "manual_constraint"
        elif link_mode == "auto_edge_path":
            connection_mode = "auto_gap"
        else:
            connection_mode = "linked"
        props = {"line": l["type"],
                 "line_cn": "坡顶线" if l["type"] == "crest" else "坡脚线",
                 "length_m": round(l["length"], 2),
                 "z_m": round(l["z"], 2),
                 "link_mode": link_mode,
                 "connection_mode": connection_mode,
                 "confidence": round(confidence, 2),
                 "source": l.get("_source", "platform_boundary"),
                 "is_inferred": bool(link_mode == "geometry_inferred"),
                 "is_manual_constraint": bool(link_mode == "manual_guided"),
                 "is_auto_gap": bool(link_mode == "auto_edge_path")}
        if link_mode == "manual_guided":
            props.update({
                "constraint_id": l.get("_constraint_id", ""),
                "anchor_line_start_index": l.get("_anchor_line_start"),
                "anchor_line_end_index": l.get("_anchor_line_end"),
                "anchor_gap_m": round(float(l.get("_anchor_gap_m", 0.0)), 2),
                "anchor_dz_m": round(float(l.get("_anchor_dz_m", 0.0)), 2),
                "guide_length_m": round(float(l.get("_guide_length_m", 0.0)), 2),
            })
        if link_mode == "auto_edge_path":
            props.update({
                "auto_gap_method": l.get("_auto_gap_method", "minimum_cost_step_edge_path"),
                "evidence_class": l.get("_evidence_class", "strong"),
                "anchor_line_start_index": l.get("_anchor_line_start"),
                "anchor_line_end_index": l.get("_anchor_line_end"),
                "anchor_gap_m": round(float(l.get("_anchor_gap_m", 0.0)), 2),
                "path_length_m": round(float(l.get("_path_length_m", l.get("length", 0.0))), 2),
                "edge_support_2m": round(float(l.get("_edge_support_2m", 0.0)), 4),
                "path_novelty_fraction": round(float(l.get("_path_novelty_fraction", 0.0)), 4),
                "length_ratio": round(float(l.get("_length_ratio", 0.0)), 4),
            })
        if crs == "ENU":
            coords = [[round(float(a), 3), round(float(b), 3)] for a, b in l["pts"]]
        else:
            coords = []
            for a, b in l["pts"]:
                lat, lon, _ = enu_to_geodetic(a, b, origin)
                coords.append([round(lon, 7), round(lat, 7)])
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString", "coordinates": coords}})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False, indent=1)

def enu_to_geodetic(e, n, origin):
    lat0 = math.radians(origin["lat"]); lon0 = math.radians(origin["lon"])
    h0 = origin.get("h", 0.0)
    A = 6378137.0; F = 1.0/298.257223563; E2 = F*(2-F)
    def llh2ecef(lat, lon, h):
        N = A/math.sqrt(1-E2*math.sin(lat)**2)
        return np.array([(N+h)*math.cos(lat)*math.cos(lon),
                         (N+h)*math.cos(lat)*math.sin(lon),
                         (N*(1-E2)+h)*math.sin(lat)])
    o = llh2ecef(lat0, lon0, h0)
    sl, cl = math.sin(lat0), math.cos(lat0)
    so, co = math.sin(lon0), math.cos(lon0)
    R = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])
    p = o + R @ np.array([e, n, 0.0])
    x, y, z = p
    lon = math.atan2(y, x)
    px = math.hypot(x, y)
    lat = math.atan2(z, px*(1-E2))
    for _ in range(8):
        N = A/math.sqrt(1-E2*math.sin(lat)**2)
        h = px/math.cos(lat)-N
        lat = math.atan2(z, px*(1-E2*N/(N+h)))
    return math.degrees(lat), math.degrees(lon), h

# ----------------------------------------------------------------------------
# 7. preview figure
# ----------------------------------------------------------------------------
def make_preview(g, lines, out_path, data_region, show_enhanced_links=False,
                 line_width=2.2, halo_width=4.4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ny, nx = g.rgb.shape[:2]
    rgb8 = np.clip(g.rgb/255.0, 0, 1).copy()
    rgb8[~data_region] = 0.6
    ext = (g.xmin, g.xmin+nx*g.res, g.ymax-ny*g.res, g.ymax)
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.imshow(rgb8, extent=ext, origin="upper")
    for l in lines:
        mode = l.get("_link_mode", "observed")
        if mode == "auto_edge_path":
            color = "#00d6c9"
            line_style = "--"
            width = max(2.0, line_width * 1.45)
        elif show_enhanced_links and mode == "geometry_inferred":
            color = "#ffb000"
            line_style = "--"
            width = max(1.5, line_width * 0.75)
        elif show_enhanced_links and mode.startswith("edge_supported"):
            color = "#35d04b"
            line_style = "--"
            width = max(1.6, line_width)
        else:
            color = "#ff3b30" if l["type"] == "crest" else "#00a2ff"
            line_style = "-"
            width = line_width
        if halo_width > 0:
            ax.plot(l["pts"][:, 0], l["pts"][:, 1], color="white",
                    alpha=0.78, lw=max(halo_width, width + 1.0),
                    ls=line_style, solid_capstyle="round", zorder=3)
        ax.plot(l["pts"][:, 0], l["pts"][:, 1], color=color,
                lw=width, ls=line_style, solid_capstyle="round", zorder=4)
    nc = sum(1 for l in lines if l["type"] == "crest")
    nt = sum(1 for l in lines if l["type"] == "toe")
    ni = sum(1 for l in lines if l.get("_link_mode") == "geometry_inferred")
    na = sum(1 for l in lines if l.get("_link_mode") == "auto_edge_path")
    if show_enhanced_links:
        ax.set_title(f"crest(red)={nc} toe(blue)={nt} inferred(orange-dash)={ni} auto-gap(cyan-dash)={na} @{g.res}m")
    else:
        ax.set_title(f"crest(red)={nc} toe(blue)={nt} auto-gap(cyan-dash)={na} @{g.res}m")
    ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
# ----------------------------------------------------------------------------
# 8. top-level run
# ----------------------------------------------------------------------------
@dataclass
class Config:
    input: str
    output: str
    grid_res_m: float = 1.0
    platform_slope_deg: float = 10.0
    face_slope_deg: float = 32.0
    min_step_drop_m: float = 2.0
    min_face_height_m: float = 3.0
    min_line_m: float = 25.0
    min_platform_px: int = 80
    min_arc_px: int = 8
    edge_seed_enabled: bool = True
    edge_seed_regions: list = field(default_factory=list)
    # Optional per-region overrides: same length/order as edge_seed_regions.
    # Example: [null, {"center_slope_deg": 28, "drop_m": 1.5, "min_px": 4}]
    edge_seed_region_params: list = field(default_factory=list)
    edge_seed_protect_links: bool = False
    edge_seed_protected_region_indices: list = field(default_factory=list)
    # User-guided connections. Each item supplies a type, an ENU guide polyline,
    # and endpoint tolerance/height gates. These links are applied after normal validation.
    guided_links: list = field(default_factory=list)
    guided_links_file: str = ""
    # Automatic gap recovery. It uses only point-cloud step edges and existing
    # automatic lines; manual guide coordinates are never consumed.
    auto_gap_enabled: bool = False
    auto_gap_min_m: float = 25.0
    auto_gap_max_m: float = 120.0
    auto_gap_sample_step_m: float = 5.0
    auto_gap_edge_drop_m: float = 1.5
    auto_gap_edge_center_slope_deg: float = 32.0
    auto_gap_edge_support_2m_min: float = 0.75
    auto_gap_novelty_min: float = 0.75
    auto_gap_length_ratio_max: float = 1.50
    auto_gap_weak_edge_support_2m_min: float = 0.70
    auto_gap_weak_novelty_min: float = 0.65
    auto_gap_nms_radius_m: float = 20.0
    auto_gap_max_bridges: int = 60
    auto_gap_max_path_candidates: int = 2000

    edge_seed_center_slope_deg: float = 25.0
    edge_seed_drop_m: float = 2.0
    edge_seed_min_line_m: float = 15.0
    edge_seed_min_px: int = 8
    edge_seed_dedup_dist_m: float = 4.0
    edge_seed_local_link_enabled: bool = True
    edge_seed_local_link_buffer_m: float = 25.0
    edge_seed_local_link_gap_m: float = 50.0
    edge_seed_local_link_dz_max_m: float = 6.0
    edge_seed_local_link_min_cos: float = 0.0
    edge_seed_local_link_lateral_max_m: float = 25.0
    multiscale_enabled: bool = True
    multiscale_slope_deg: float = 12.0
    multiscale_min_platform_px: int = 50
    multiscale_min_line_m: float = 20.0
    multiscale_dedup_dist_m: float = 4.0
    multiscale_drop_m: float = 2.0
    multiscale_edge_center_slope_deg: float = 25.0
    multiscale_edge_dist_m: float = 2.0
    multiscale_edge_min_fraction: float = 0.60
    multiscale_network_gap_m: float = 60.0
    multiscale_isolated_min_length_m: float = 30.0
    multiscale_isolated_edge_fraction: float = 0.80
    simplify_tol_m: float = 1.2
    smooth_sigma_cells: float = 1.0
    ray_m: list = field(default_factory=lambda: [2.0, 3.0, 4.0, 5.0])
    output_crs: str = "WGS84"     # ENU | WGS84 | BOTH
    link_gap_m: float = 0.0       # 0 = off; >0 enables fragment linking
    link_dz_max_m: float = 5.0
    link_min_cos: float = 0.35
    link_lateral_max_m: float = 15.0
    link_w_dist: float = 4.0
    link_w_z: float = 1.0
    link_passes: int = 4

    # Enhanced gap recovery.  It keeps the original one-to-one matching but
    # adds direct step-edge support and a separate geometry-inferred layer.
    enhanced_link: bool = True
    link_short_gap_m: float = 25.0
    link_short_dz_max_m: float = 3.0
    link_short_lateral_max_m: float = 18.0
    link_infer_gap_m: float = 100.0
    link_infer_dz_max_m: float = 9.0
    link_infer_min_cos: float = 0.10
    link_infer_lateral_max_m: float = 18.0
    link_edge_dist_m: float = 2.0
    link_edge_min_fraction: float = 0.70
    link_w_edge: float = 2.5
    link_allow_inferred: bool = True
    link_max_candidates_per_endpoint: int = 6

    preview: bool = True
    preview_show_enhanced_links: bool = False
    preview_line_width: float = 1.2
    preview_halo_width: float = 0.0
    enu_origin: dict = field(default_factory=lambda: {
        "lat": 25.66087416, "lon": 113.34421763, "h": 507.004})

    @classmethod
    def from_dict(cls, d, base_dir):
        c = cls(input=d.get("input", ""), output=d.get("output", ""))
        for k in ("grid_res_m", "platform_slope_deg", "face_slope_deg",
                  "min_step_drop_m", "min_face_height_m", "min_line_m",
                  "min_platform_px", "min_arc_px",
                  "edge_seed_enabled", "edge_seed_regions",
                  "edge_seed_region_params",
                  "edge_seed_protect_links", "edge_seed_protected_region_indices",
                  "guided_links", "guided_links_file",
                  "auto_gap_enabled", "auto_gap_min_m", "auto_gap_max_m",
                  "auto_gap_sample_step_m", "auto_gap_edge_drop_m",
                  "auto_gap_edge_center_slope_deg",
                  "auto_gap_edge_support_2m_min", "auto_gap_novelty_min",
                  "auto_gap_length_ratio_max",
                  "auto_gap_weak_edge_support_2m_min",
                  "auto_gap_weak_novelty_min", "auto_gap_nms_radius_m",
                  "auto_gap_max_bridges", "auto_gap_max_path_candidates",
                  "auto_gap_enabled", "auto_gap_min_m", "auto_gap_max_m",
                "auto_gap_sample_step_m", "auto_gap_edge_drop_m",
                "auto_gap_edge_center_slope_deg",
                "auto_gap_edge_support_2m_min", "auto_gap_novelty_min",
                "auto_gap_length_ratio_max",
                "auto_gap_weak_edge_support_2m_min",
                "auto_gap_weak_novelty_min", "auto_gap_nms_radius_m",
                "auto_gap_max_bridges", "auto_gap_max_path_candidates",
                "edge_seed_center_slope_deg",
                  "edge_seed_drop_m", "edge_seed_min_line_m",
                  "edge_seed_min_px", "edge_seed_dedup_dist_m",
                "edge_seed_local_link_enabled", "edge_seed_local_link_buffer_m",
                "edge_seed_local_link_gap_m", "edge_seed_local_link_dz_max_m",
                "edge_seed_local_link_min_cos",
                "edge_seed_local_link_lateral_max_m",
                  "edge_seed_local_link_enabled", "edge_seed_local_link_buffer_m",
                  "edge_seed_local_link_gap_m", "edge_seed_local_link_dz_max_m",
                  "edge_seed_local_link_min_cos",
                  "edge_seed_local_link_lateral_max_m",
                  "multiscale_enabled", "multiscale_slope_deg",
                  "multiscale_min_platform_px", "multiscale_min_line_m",
                  "multiscale_dedup_dist_m", "multiscale_drop_m",
                  "multiscale_edge_center_slope_deg",
                  "multiscale_edge_dist_m", "multiscale_edge_min_fraction",
                  "multiscale_network_gap_m",
                  "multiscale_isolated_min_length_m",
                  "multiscale_isolated_edge_fraction",
                  "simplify_tol_m",
                  "smooth_sigma_cells", "ray_m", "output_crs", "preview",
                  "enu_origin", "link_gap_m", "link_dz_max_m", "link_min_cos",
                  "link_lateral_max_m", "link_w_dist", "link_w_z", "link_passes",
                  "enhanced_link", "link_short_gap_m", "link_short_dz_max_m",
                  "link_short_lateral_max_m", "link_infer_gap_m",
                  "link_infer_dz_max_m", "link_infer_min_cos",
                  "link_infer_lateral_max_m", "link_edge_dist_m",
                  "link_edge_min_fraction", "link_w_edge",
                  "link_allow_inferred", "link_max_candidates_per_endpoint",
                  "preview_show_enhanced_links", "preview_line_width",
                  "preview_halo_width"):
            if k in d:
                setattr(c, k, d[k])
        if not os.path.isabs(c.input):
            c.input = os.path.join(base_dir, c.input)
        if not os.path.isabs(c.output):
            c.output = os.path.join(base_dir, c.output)
        if c.guided_links_file:
            if not os.path.isabs(c.guided_links_file):
                c.guided_links_file = os.path.join(base_dir, c.guided_links_file)
            with open(c.guided_links_file, "r", encoding="utf-8") as handle:
                c.guided_links = json.load(handle)
        return c

def run(cfg: Config, verbose=True):
    os.makedirs(cfg.output, exist_ok=True)
    g = build_grid(cfg.input, cfg.grid_res_m, verbose=verbose)
    dsm, slope, data_region = prepare_dsm(g, cfg.smooth_sigma_cells, verbose=verbose)
    lines = extract_lines(dsm, slope, data_region, cfg, g.xmin, g.ymax, verbose=verbose)
    lines = augment_multiscale_lines(
        lines, dsm, data_region, cfg, g.xmin, g.ymax, verbose=verbose)
    if cfg.link_gap_m > 0:
        if cfg.enhanced_link:
            lines = link_lines_enhanced(lines, dsm, g, cfg, verbose=verbose)
        else:
            lines = link_lines(lines, dsm, g, cfg, verbose=verbose)
    lines = link_lines_local(lines, dsm, g, cfg, verbose=verbose)
    lines = validate_lines(lines, dsm, g, cfg.min_face_height_m, verbose=verbose)
    lines = recover_auto_gap_bridges(lines, dsm, g, cfg, verbose=verbose)
    lines = apply_guided_links(lines, cfg, verbose=verbose)
    base_lines = [line for line in lines if line.get("_link_mode") != "auto_edge_path"]
    auto_gap_lines = [line for line in lines if line.get("_link_mode") == "auto_edge_path"]

    crs_set = {"ENU", "WGS84", "BOTH"}
    crs = cfg.output_crs.upper()
    if crs not in crs_set:
        raise ValueError("output_crs must be ENU/WGS84/BOTH")
    if crs in ("ENU", "BOTH"):
        export_geojson(lines, os.path.join(cfg.output, "slope_lines_ENU.json"),
                       "ENU", cfg.enu_origin)
        export_geojson(base_lines, os.path.join(cfg.output, "slope_lines_base_ENU.json"),
                       "ENU", cfg.enu_origin)
        export_geojson(auto_gap_lines, os.path.join(cfg.output, "slope_lines_auto_gap_only_ENU.json"),
                       "ENU", cfg.enu_origin)
    if crs in ("WGS84", "BOTH"):
        export_geojson(lines, os.path.join(cfg.output, "slope_lines_WGS84.json"),
                       "WGS84", cfg.enu_origin)
        export_geojson(base_lines, os.path.join(cfg.output, "slope_lines_base_WGS84.json"),
                       "WGS84", cfg.enu_origin)
        export_geojson(auto_gap_lines, os.path.join(cfg.output, "slope_lines_auto_gap_only_WGS84.json"),
                       "WGS84", cfg.enu_origin)
    if cfg.preview:
        make_preview(g, lines, os.path.join(cfg.output, "preview.png"),
                     data_region,
                     show_enhanced_links=cfg.preview_show_enhanced_links,
                     line_width=cfg.preview_line_width,
                     halo_width=cfg.preview_halo_width)
    # sidecar params
    meta = {"input": os.path.abspath(cfg.input), "output": os.path.abspath(cfg.output),
            "params": {k: getattr(cfg, k) for k in (
                "grid_res_m", "platform_slope_deg", "face_slope_deg",
                "min_step_drop_m", "min_face_height_m", "min_line_m",
                "min_platform_px", "min_arc_px",
                "edge_seed_enabled", "edge_seed_regions",
                "edge_seed_region_params",
                "edge_seed_protect_links", "edge_seed_protected_region_indices",
                "guided_links", "guided_links_file",
                "auto_gap_enabled", "auto_gap_min_m", "auto_gap_max_m",
                "auto_gap_sample_step_m", "auto_gap_edge_drop_m",
                "auto_gap_edge_center_slope_deg",
                "auto_gap_edge_support_2m_min", "auto_gap_novelty_min",
                "auto_gap_length_ratio_max",
                "auto_gap_weak_edge_support_2m_min",
                "auto_gap_weak_novelty_min", "auto_gap_nms_radius_m",
                "auto_gap_max_bridges", "auto_gap_max_path_candidates",
                "edge_seed_center_slope_deg",
                "edge_seed_drop_m", "edge_seed_min_line_m",
                "edge_seed_min_px", "edge_seed_dedup_dist_m",
                "edge_seed_local_link_enabled", "edge_seed_local_link_buffer_m",
                "edge_seed_local_link_gap_m", "edge_seed_local_link_dz_max_m",
                "edge_seed_local_link_min_cos",
                "edge_seed_local_link_lateral_max_m",
                "multiscale_enabled", "multiscale_slope_deg",
                "multiscale_min_platform_px", "multiscale_min_line_m",
                "multiscale_dedup_dist_m", "multiscale_drop_m",
                "multiscale_edge_center_slope_deg",
                "multiscale_edge_dist_m",
                "multiscale_edge_min_fraction",
                "multiscale_network_gap_m",
                "multiscale_isolated_min_length_m",
                "multiscale_isolated_edge_fraction",
                "simplify_tol_m",
                "smooth_sigma_cells", "ray_m", "output_crs",
                "enhanced_link", "link_gap_m", "link_dz_max_m",
                "link_min_cos", "link_lateral_max_m", "link_passes",
                "link_short_gap_m", "link_short_dz_max_m",
                "link_short_lateral_max_m", "link_infer_gap_m",
                "link_infer_dz_max_m", "link_infer_min_cos",
                "link_infer_lateral_max_m", "link_edge_dist_m",
                "link_edge_min_fraction", "link_w_edge",
                "link_allow_inferred", "link_max_candidates_per_endpoint",
                "preview_show_enhanced_links", "preview_line_width",
                "preview_halo_width")},
            "enu_origin": cfg.enu_origin,
            "line_counts": {"crest": sum(1 for l in lines if l["type"] == "crest"),
                            "toe": sum(1 for l in lines if l["type"] == "toe"),
                            "inferred": sum(1 for l in lines if l.get("_link_mode") == "geometry_inferred"),
                            "edge_supported": sum(1 for l in lines if str(l.get("_link_mode", "")).startswith("edge_supported")),
                            "edge_seed": sum(1 for l in lines if str(l.get("_source", "")).startswith("edge_seed")),
                            "multiscale_seed": sum(1 for l in lines if str(l.get("_source", "")).startswith("multiscale_seed")),
                            "manual_constraint": sum(1 for l in lines if l.get("_source") == "manual_constraint"),
                            "auto_gap": sum(1 for l in lines if l.get("_link_mode") == "auto_edge_path")},
            "total_length_m": round(sum(l["length"] for l in lines), 2)}
    with open(os.path.join(cfg.output, "run_params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[done] {meta['line_counts']}  -> {os.path.abspath(cfg.output)}")
    return meta
