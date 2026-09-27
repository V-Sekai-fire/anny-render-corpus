"""MoGe-3 depth agreement: the second mask scorer of plan step 2 (3b).

A candidate person mask is plausible when its boundary runs along a depth
discontinuity. MoGe-3 (`Ruicheng/moge-3-vitl`, MIT) supplies the depth; this file
measures how well a binary mask's boundary follows it, and, for renders, how far
MoGe-3's depth can be trusted against a Mitsuba reference depth.

Subcommands
-----------
infer    run MoGe-3 on frames, write data/moge/<id>.{depth.npy,normal.npy,mask.png,json}
score    score RF-DETR person candidates from dress-on's results.bin -> JSONL
compare  MoGe depth vs a reference depth .exr (median-aligned AbsRel, delta<1.25)
sheet    contact sheet: frame + MoGe depth, top candidate outline, its score

Mask score (`mask_depth_agreement`)
    edges    = |grad d| / d > EDGE_REL (i.e. |grad log d|, per pixel) OR the
               boundary of MoGe's own validity mask
    edge_frac      fraction of the mask's 1-px inner boundary within EDGE_TOL (2) px of an edge
    edge_frac_cell same, tolerance half a 78x78 mask cell (0.5*max(H,W)/78 px)
    edge_dist_mean mean boundary-to-nearest-edge distance in px, capped at SEARCH (25)
    gap            per boundary pixel with an edge within SEARCH px: median log-depth
                   8..20 px outside minus inside that edge along the mask's outward
                   normal; median of exp(gap)-1, so +0.3 means "outside is 30% farther".

Runtime: `infer` needs the MoGe-3 env (torch + moge + FlexGEMM/Triton), which is
C:/Users/ernest.lee/AppData/Local/moge3/pixi.toml on the Windows 4090; it is not in
3-interactor/moge-upstream because that checkout is pinned at a bare upstream commit.
The other subcommands need only numpy, opencv, pillow and matplotlib (and OpenEXR for
`compare`), all of which that env has.
"""
import argparse
import json
import os
import time

import numpy as np

EDGE_REL = 0.04     # |grad log d| per pixel that counts as a depth edge
EDGE_TOL = 2.0      # px: a boundary pixel within this of an edge "agrees"
SEARCH = 25.0       # px: how far from a mask-boundary pixel to look for its depth edge
BAND = (8, 20)      # px: inside/outside band either side of that depth edge

Q, C, M = 100, 91, 78
REC = Q * 4 + Q * C + Q * M * M

HERE = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    DESK = r"C:\Users\ernest.lee\Desktop\frameeyeosc-contact-sheets"
else:
    DESK = "/mnt/c/Users/ernest.lee/Desktop/frameeyeosc-contact-sheets"
RFD = os.path.join(HERE, "..", "..", "3-interactor", "dress-on", "build", "rfdetr_frames")
OUT = os.path.join(HERE, "data", "moge")


def frame_ids(frames_txt):
    ids = []
    for line in open(frames_txt):
        line = line.strip()
        if line:
            base = line.replace("\\", "/").rsplit("/", 1)[-1]
            ids.append(base.rsplit(".", 1)[0])
    return ids


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------- MoGe-3 inference

def cmd_infer(a):
    import torch
    from moge.model.v3 import MoGeModel
    import cv2

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0")
    print("device", torch.cuda.get_device_name(dev), flush=True)
    model = MoGeModel.from_pretrained(a.model).to(dev).eval()
    ids = frame_ids(a.frames_txt)
    log = []
    for i, fid in enumerate(ids):
        rgb = load_rgb(os.path.join(a.frames_dir, fid + ".png"))
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model.infer(x, refine_steps=a.refine_steps, use_fp16=a.fp16)
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        depth = o["depth"].float().cpu().numpy().astype(np.float32)
        mask = o["mask"].cpu().numpy().astype(bool)
        K = o["intrinsics"].float().cpu().numpy()
        depth[~mask | ~np.isfinite(depth)] = np.nan
        np.save(os.path.join(a.out, fid + ".depth.npy"), depth)
        if "normal" in o and o["normal"] is not None:
            nrm = o["normal"].float().cpu().numpy().astype(np.float16)
            np.save(os.path.join(a.out, fid + ".normal.npy"), nrm)
        cv2.imwrite(os.path.join(a.out, fid + ".mask.png"), mask.astype(np.uint8) * 255)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        meta = dict(id=fid, h=int(depth.shape[0]), w=int(depth.shape[1]),
                    intrinsics_normalized=K.tolist(),
                    fov_x_deg=float(np.degrees(2 * np.arctan(0.5 / fx))),
                    fov_y_deg=float(np.degrees(2 * np.arctan(0.5 / fy))),
                    valid_frac=float(mask.mean()),
                    depth_median=float(np.nanmedian(depth)) if mask.any() else None,
                    seconds=dt, peak_vram_gib=peak, model=a.model,
                    refine_steps=a.refine_steps, fp16=a.fp16)
        json.dump(meta, open(os.path.join(a.out, fid + ".json"), "w"), indent=1)
        log.append(meta)
        print("%2d/%d %s  %.2fs  %.2f GiB  fov_x %.1f  valid %.3f" % (
            i + 1, len(ids), fid, dt, peak, meta["fov_x_deg"], meta["valid_frac"]), flush=True)
    json.dump(log, open(os.path.join(a.out, "infer_log.json"), "w"), indent=1)


# ---------------------------------------------------------------- depth edges

def _filled_logdepth(depth):
    """log depth with invalid pixels set to log(1.5 * max valid): 'far', not missing."""
    valid = np.isfinite(depth) & (depth > 0)
    far = 1.5 * np.nanmax(np.where(valid, depth, np.nan)) if valid.any() else 1.0
    d = np.where(valid, depth, far).astype(np.float32)
    return np.log(d), valid


def depth_edges(depth, edge_rel=EDGE_REL):
    import cv2
    ld, valid = _filled_logdepth(depth)
    gy, gx = np.gradient(ld)
    g = np.hypot(gx, gy)            # = |grad d| / d
    edges = g > edge_rel
    v8 = valid.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    vb = (cv2.dilate(v8, k) != cv2.erode(v8, k))
    return edges | vb, g, ld, valid


def edge_distance(edges):
    """Distance from every pixel to the nearest depth-edge pixel, and that pixel's (y, x)."""
    from scipy.ndimage import distance_transform_edt
    dist, (iy, ix) = distance_transform_edt(~edges, return_indices=True)
    return dist.astype(np.float32), iy, ix


def mask_depth_agreement(depth, mask, edges=None, ld=None, edt=None, cell_tol=None):
    import cv2
    if edges is None or ld is None:
        edges, _, ld, _ = depth_edges(depth)
    if edt is None:
        edt = edge_distance(edges)
    dist, iy, ix = edt
    h, w = mask.shape
    if cell_tol is None:
        cell_tol = 0.5 * max(h, w) / M
    m8 = mask.astype(np.uint8)
    if m8.sum() == 0:
        return dict(edge_frac=0.0, edge_frac_cell=0.0, cell_tol_px=cell_tol, edge_dist_mean=None,
                    edge_found_frac=0.0, gap_rel=None, gap_abs=None, n_boundary=0)
    k = np.ones((3, 3), np.uint8)
    boundary = (m8 == 1) & (cv2.erode(m8, k, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
    by, bx = np.nonzero(boundary)
    db = dist[by, bx]
    # outward mask normal at each boundary pixel, from the smoothed mask's gradient
    sm = cv2.GaussianBlur(m8.astype(np.float32), (0, 0), 3.0)
    gy, gx = np.gradient(sm)
    ny, nx = -gy[by, bx], -gx[by, bx]
    nn = np.hypot(ny, nx)
    found = (db <= SEARCH) & (nn > 1e-6)
    ey, ex = iy[by, bx][found], ix[by, bx][found]
    ny, nx = ny[found] / nn[found], nx[found] / nn[found]
    t = np.arange(BAND[0], BAND[1] + 1, dtype=np.float32)

    def side(sign):
        yy = np.clip(np.rint(ey[:, None] + sign * t[None] * ny[:, None]), 0, h - 1).astype(int)
        xx = np.clip(np.rint(ex[:, None] + sign * t[None] * nx[:, None]), 0, w - 1).astype(int)
        return np.median(ld[yy, xx], axis=1)

    if found.any():
        gap = side(+1) - side(-1)
        gap_rel = float(np.median(np.expm1(gap)))
        gap_abs = float(np.median(np.abs(np.expm1(gap))))
    else:
        gap_rel = gap_abs = None
    return dict(edge_frac=float((db <= EDGE_TOL).mean()),
                edge_frac_cell=float((db <= cell_tol).mean()), cell_tol_px=float(cell_tol),
                edge_dist_mean=float(np.minimum(db, SEARCH).mean()),
                edge_found_frac=float(found.mean()),
                gap_rel=gap_rel, gap_abs=gap_abs, n_boundary=int(by.size))


# ---------------------------------------------------------------- RF-DETR candidates

def read_results(path, n):
    raw = np.fromfile(path, dtype=np.float32)
    assert raw.size == n * REC, (raw.size, n * REC)
    out = []
    for i in range(n):
        r = raw[i * REC:(i + 1) * REC]
        boxes = r[:Q * 4].reshape(Q, 4)
        logits = r[Q * 4:Q * 4 + Q * C].reshape(Q, C)
        masks = r[Q * 4 + Q * C:].reshape(Q, M, M)
        out.append((boxes, logits, masks))
    return out


def candidates(boxes, logits, masks, h, w, k=3):
    import cv2
    p = 1 / (1 + np.exp(-logits[:, 1]))
    area = boxes[:, 2] * boxes[:, 3]
    rank = np.argsort(-(p * area))[:k]
    res = []
    for r, q in enumerate(rank):
        up = cv2.resize(masks[q], (w, h), interpolation=cv2.INTER_LINEAR)
        res.append(dict(rank=r, query=int(q), p_person=float(p[q]), box=boxes[q].tolist(),
                        rank_key=float(p[q] * area[q]), mask=up > 0.0))  # sigmoid>0.5 <=> logit>0
    return res


def corruptions(mask):
    """Deliberately worse masks, a control: the scorer should rank these below the source."""
    import cv2
    m8 = mask.astype(np.uint8)
    h, w = mask.shape
    s = max(4, int(0.02 * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    T = np.float32([[1, 0, 2 * s], [0, 1, s]])
    return dict(eroded=cv2.erode(m8, k) > 0, dilated=cv2.dilate(m8, k) > 0,
                shifted=cv2.warpAffine(m8, T, (w, h), flags=cv2.INTER_NEAREST) > 0)


def cmd_score(a):
    ids = frame_ids(a.frames_txt)
    res = read_results(a.results, len(ids))
    with open(a.jsonl, "w") as f:
        for fid, (boxes, logits, masks) in zip(ids, res):
            depth = np.load(os.path.join(a.moge, fid + ".depth.npy"))
            h, w = depth.shape
            edges, _, ld, _ = depth_edges(depth)
            edt = edge_distance(edges)
            for c in candidates(boxes, logits, masks, h, w):
                s = mask_depth_agreement(depth, c["mask"], edges, ld, edt)
                rec = dict(frame=fid, variant="candidate", rank=c["rank"], query=c["query"],
                           p_person=c["p_person"], box=c["box"], rank_key=c["rank_key"],
                           mask_frac=float(c["mask"].mean()), **s)
                f.write(json.dumps(rec) + "\n")
                if c["rank"] == 0:
                    for name, cm in corruptions(c["mask"]).items():
                        s2 = mask_depth_agreement(depth, cm, edges, ld, edt)
                        f.write(json.dumps(dict(frame=fid, variant=name, rank=0, query=c["query"],
                                                mask_frac=float(cm.mean()), **s2)) + "\n")
            print(fid, flush=True)


# ---------------------------------------------------------------- MoGe vs reference depth

def read_exr_depth(path):
    import OpenEXR
    with OpenEXR.File(path) as f:
        ch = f.channels()
        for name in ("Z", "depth", "D", "Y", "R", "RGB", "RGBA"):
            if name in ch:
                px = ch[name].pixels
                return (px if px.ndim == 2 else px[..., 0]).astype(np.float32)
        px = next(iter(ch.values())).pixels
        return (px if px.ndim == 2 else px[..., 0]).astype(np.float32)


def depth_metrics(pred, ref, mask=None):
    """AbsRel and delta<1.25 after median scale alignment, and after LSQ scale+shift."""
    ok = np.isfinite(pred) & np.isfinite(ref) & (pred > 0) & (ref > 0)
    if mask is not None:
        ok &= mask
    p, r = pred[ok].astype(np.float64), ref[ok].astype(np.float64)
    if p.size < 16:
        return dict(n=int(p.size))
    out = dict(n=int(p.size))
    pm = p * (np.median(r) / np.median(p))
    A = np.stack([p, np.ones_like(p)], 1)
    sc, sh = np.linalg.lstsq(A, r, rcond=None)[0]
    pa = np.maximum(sc * p + sh, 1e-6)
    for tag, q in (("median", pm), ("affine", pa)):
        out["absrel_" + tag] = float(np.mean(np.abs(q - r) / r))
        out["delta125_" + tag] = float(np.mean(np.maximum(q / r, r / q) < 1.25))
    out["scale"], out["shift"] = float(sc), float(sh)
    return out


def cmd_compare(a):
    pred = np.load(a.moge) if a.moge.endswith(".npy") else read_exr_depth(a.moge)
    ref = read_exr_depth(a.ref) if a.ref.endswith(".exr") else np.load(a.ref)
    if ref.shape != pred.shape:
        import cv2
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = None
    if a.mask:
        from PIL import Image
        mask = np.asarray(Image.open(a.mask).convert("L")) > 127
    print(json.dumps(depth_metrics(pred, ref, mask)))


# ---------------------------------------------------------------- contact sheet

def cmd_sheet(a):
    import cv2
    import matplotlib
    from PIL import Image, ImageDraw
    cmap = matplotlib.colormaps["turbo"]
    ids = frame_ids(a.frames_txt)
    res = read_results(a.results, len(ids))
    rows = {}
    for line in open(a.jsonl):
        r = json.loads(line)
        rows.setdefault(r["frame"], []).append(r)
    TW = a.tile_w
    tiles = []
    for fid, (boxes, logits, masks) in zip(ids, res):
        depth = np.load(os.path.join(a.moge, fid + ".depth.npy"))
        h, w = depth.shape
        rgb = load_rgb(os.path.join(a.frames_dir, fid + ".png"))
        c0 = candidates(boxes, logits, masks, h, w, k=1)[0]
        valid = np.isfinite(depth)
        inv = np.where(valid, 1.0 / np.where(valid, depth, 1), 0)
        lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0, 1)
        col = (cmap(np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)
        col[~valid] = 40
        th = int(round(h * TW / w))
        m8 = cv2.resize(c0["mask"].astype(np.uint8), (TW, th), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pair = []
        for img in (rgb, col):
            t = np.ascontiguousarray(cv2.resize(img, (TW, th), interpolation=cv2.INTER_AREA))
            cv2.drawContours(t, cnts, -1, (255, 255, 255), 3)
            cv2.drawContours(t, cnts, -1, (255, 0, 160), 1)
            pair.append(t)
        tile = Image.fromarray(np.concatenate(pair, 1))
        d = ImageDraw.Draw(tile)
        rr = {(r["variant"], r.get("rank")): r for r in rows.get(fid, [])}
        top = rr.get(("candidate", 0), {})
        others = " ".join("#%d %.2f" % (k, rr[("candidate", k)]["edge_frac_cell"])
                          for k in (1, 2) if ("candidate", k) in rr)
        ctrl = " ".join("%s %.2f" % (v[:3], rr[(v, 0)]["edge_frac_cell"])
                        for v in ("eroded", "dilated", "shifted") if (v, 0) in rr)
        g = top.get("gap_rel")
        txt = "%s p=%.2f e2=%.2f ecell=%.2f d=%.1fpx gap=%s | %s | ctrl %s" % (
            fid, top.get("p_person", 0), top.get("edge_frac", 0), top.get("edge_frac_cell", 0),
            top.get("edge_dist_mean") or 0, "%+.2f" % g if g is not None else "-", others, ctrl)
        d.rectangle([0, 0, tile.width, 16], fill=(0, 0, 0))
        d.text((4, 2), txt, fill=(255, 255, 255))
        tiles.append(tile)
    cols = a.cols
    tw, th = tiles[0].size
    nrows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * 4, nrows * th + (nrows - 1) * 4), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (tw + 4), (i // cols) * (th + 4)))
    sheet.save(a.out)
    print(a.out, sheet.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    frames_txt = os.path.join(RFD, "frames.txt")
    frames_dir = os.path.join(DESK, "frames")
    jsonl = os.path.join(OUT, "rfdetr_depth_scores.jsonl")

    p = sp.add_parser("infer")
    p.add_argument("--frames-txt", default=frames_txt)
    p.add_argument("--frames-dir", default=frames_dir)
    p.add_argument("--out", default=OUT)
    p.add_argument("--model", default="Ruicheng/moge-3-vitl")
    p.add_argument("--refine-steps", type=int, default=3)
    p.add_argument("--fp16", action="store_true")

    for name in ("score", "sheet"):
        p = sp.add_parser(name)
        p.add_argument("--frames-txt", default=frames_txt)
        p.add_argument("--results", default=os.path.join(RFD, "results.bin"))
        p.add_argument("--moge", default=OUT)
        p.add_argument("--jsonl", default=jsonl)
        if name == "sheet":
            p.add_argument("--frames-dir", default=frames_dir)
            p.add_argument("--out", default=os.path.join(DESK, "moge3-depth-23.png"))
            p.add_argument("--tile-w", type=int, default=400)
            p.add_argument("--cols", type=int, default=3)

    p = sp.add_parser("compare")
    p.add_argument("--moge", required=True, help="MoGe depth .npy (or .exr)")
    p.add_argument("--ref", required=True, help="reference (Mitsuba) depth .exr (or .npy)")
    p.add_argument("--mask", help="optional PNG: evaluate only where >127")

    a = ap.parse_args()
    dict(infer=cmd_infer, score=cmd_score, compare=cmd_compare, sheet=cmd_sheet)[a.cmd](a)


if __name__ == "__main__":
    main()
