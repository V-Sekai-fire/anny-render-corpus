"""Tighten RF-DETR person masks with an HR alpha matte (plan step 2, pseudo-labels).

RF-DETR seg-nano's masks are 78x78 over the whole (squashed to 312x312) frame: about
25 px per cell at 1920 wide, upsampled bilinearly, so its outline is a smooth halo
around the character rather than the silhouette MoGe-3's depth shows (hair gaps,
arm/body gaps). BiRefNet_HR-matting (ZhengPeng7, MIT; 2048x2048, prior-free) gives
that silhouette as an alpha matte; RF-DETR picks the instance:

    region  = RF-DETR's top person candidate, dilated by 1.5 cells
    refined = the components of (alpha > 0.5) inside region that meet the
              candidate's 1-cell erosion (so another avatar's matte never joins)

Both masks are scored by depth_agreement.mask_depth_agreement against MoGe-3's depth
(data/moge/<id>.depth.npy): a tighter mask must sit closer to the depth edges.

Runtime: C:/Users/ernest.lee/AppData/Local/matting-hr/pixi.toml (torch cu130 on the
Windows 4090, transformers, timm, kornia, opencv, scipy, matplotlib).

    pixi run --manifest-path C:/Users/ernest.lee/AppData/Local/matting-hr/pixi.toml \
        python matte_refine.py            # writes data/matting/, the JSONL and the sheet
"""
import argparse
import json
import os
import time

import numpy as np

import depth_agreement as da

OUT = os.path.join(da.HERE, "data", "matting")
RES = 2048
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(name):
    import torch
    from transformers import AutoModelForImageSegmentation
    m = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True)
    m.to("cuda").eval().half()
    torch.set_float32_matmul_precision("high")
    return m


def matte(model, rgb):
    """Alpha in [0, 1] at the frame's size."""
    import cv2
    import torch
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (RES, RES), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        y = model(torch.from_numpy(x).cuda().half())[-1].sigmoid()
    a = y[0, 0].float().cpu().numpy()
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


def refine(coarse, alpha, cell):
    import cv2
    m8 = coarse.astype(np.uint8)

    def disk(r):
        r = max(1, int(round(r)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    region = cv2.dilate(m8, disk(1.5 * cell)) > 0
    seed = cv2.erode(m8, disk(cell)) > 0
    if not seed.any():
        seed = coarse
    fg = ((alpha > 0.5) & region).astype(np.uint8)
    n, lab = cv2.connectedComponents(fg, connectivity=8)
    keep = np.unique(lab[seed & (fg > 0)])
    keep = keep[keep > 0]
    return np.isin(lab, keep) & (fg > 0)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def cmd_run(a):
    import cv2
    os.makedirs(a.out, exist_ok=True)
    ids = da.frame_ids(a.frames_txt)
    res = da.read_results(a.results, len(ids))
    model = load_model(a.model)
    rows = []
    with open(a.jsonl, "w") as f:
        for fid, (boxes, logits, masks) in zip(ids, res):
            rgb = da.load_rgb(os.path.join(a.frames_dir, fid + ".png"))
            h, w = rgb.shape[:2]
            t0 = time.time()
            alpha = matte(model, rgb)
            dt = time.time() - t0
            c0 = da.candidates(boxes, logits, masks, h, w, k=1)[0]
            cell = max(h, w) / da.M
            ref = refine(c0["mask"], alpha, cell)
            np.save(os.path.join(a.out, fid + ".alpha.npy"), alpha.astype(np.float16))
            cv2.imwrite(os.path.join(a.out, fid + ".refined.png"), ref.astype(np.uint8) * 255)
            depth = np.load(os.path.join(a.moge, fid + ".depth.npy"))
            edges, _, ld, _ = da.depth_edges(depth)
            edt = da.edge_distance(edges)
            rec = dict(frame=fid, p_person=c0["p_person"], matte_s=dt, iou_coarse_refined=iou(c0["mask"], ref),
                       coarse_frac=float(c0["mask"].mean()), refined_frac=float(ref.mean()),
                       coarse=da.mask_depth_agreement(depth, c0["mask"], edges, ld, edt),
                       refined=da.mask_depth_agreement(depth, ref, edges, ld, edt))
            f.write(json.dumps(rec) + "\n")
            rows.append((fid, rgb, depth, alpha, c0["mask"], ref, rec))
            print("%s matte %.2fs iou %.2f edge@cell %.2f -> %.2f dist %.1f -> %.1f px" % (
                fid, dt, rec["iou_coarse_refined"], rec["coarse"]["edge_frac_cell"],
                rec["refined"]["edge_frac_cell"], rec["coarse"]["edge_dist_mean"] or 0,
                rec["refined"]["edge_dist_mean"] or 0), flush=True)
    sheet(rows, a.sheet, a.tile_w, a.cols)


def sheet(rows, out, tw, cols):
    import cv2
    import matplotlib
    from PIL import Image, ImageDraw
    cmap = matplotlib.colormaps["turbo"]
    tiles = []
    for fid, rgb, depth, alpha, coarse, ref, rec in rows:
        h, w = depth.shape
        th = int(round(h * tw / w))
        valid = np.isfinite(depth)
        inv = np.where(valid, 1.0 / np.where(valid, depth, 1), 0)
        lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0, 1)
        col = (cmap(np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)
        al = (np.repeat(alpha[..., None], 3, 2) * 255).astype(np.uint8)

        def contours(m):
            m8 = cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
            return cv2.findContours(m8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        cc, cr = contours(coarse), contours(ref)
        panels = []
        for img, both in ((rgb, True), (col, False), (al, False)):
            t = np.ascontiguousarray(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))
            if both:
                cv2.drawContours(t, cc, -1, (255, 0, 160), 1)
            cv2.drawContours(t, cr, -1, (0, 255, 0), 1)
            panels.append(t)
        tile = Image.fromarray(np.concatenate(panels, 1))
        d = ImageDraw.Draw(tile)
        c, r = rec["coarse"], rec["refined"]
        txt = "%s  magenta=RF-DETR ecell %.2f d %.1fpx | green=matte ecell %.2f d %.1fpx | IoU %.2f" % (
            fid, c["edge_frac_cell"], c["edge_dist_mean"] or 0, r["edge_frac_cell"], r["edge_dist_mean"] or 0,
            rec["iou_coarse_refined"])
        d.rectangle([0, 0, tile.width, 16], fill=(0, 0, 0))
        d.text((4, 2), txt, fill=(255, 255, 255))
        tiles.append(tile)
    tw2, th2 = tiles[0].size
    nrows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (cols * tw2 + (cols - 1) * 4, nrows * th2 + (nrows - 1) * 4), (20, 20, 20))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * (tw2 + 4), (i // cols) * (th2 + 4)))
    s.save(out)
    print(out, s.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames-txt", default=os.path.join(da.RFD, "frames.txt"))
    ap.add_argument("--frames-dir", default=os.path.join(da.DESK, "frames"))
    ap.add_argument("--results", default=os.path.join(da.RFD, "results.bin"))
    ap.add_argument("--moge", default=da.OUT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jsonl", default=os.path.join(OUT, "matte_refine.jsonl"))
    ap.add_argument("--sheet", default=os.path.join(da.DESK, "matting-hr-refined-23.png"))
    ap.add_argument("--model", default="ZhengPeng7/BiRefNet_HR-matting")
    ap.add_argument("--tile-w", type=int, default=360)
    ap.add_argument("--cols", type=int, default=2)
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
