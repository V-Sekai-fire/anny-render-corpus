# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Render posed VRM avatars through a textured MToon integrator, with COCO instance masks.

Each image holds 1-3 avatars from vrm_models.tsv against a CC0 HDRI (vrm_backgrounds.tsv) or
a plain gradient. Cameras come from `sphere_hammersley`, biased frontal, in three framings:
face close-up, bust and full body. Masks are per-avatar coverage AOVs from the colour pass
itself, so they agree with the antialiased edge. `--aux` adds a separate metric depth /
normal / shape_index pass that never touches the colour pass.

    python vrm_render.py --count 300 [--aux] [--spp 16] [--threads 0]
    python vrm_render.py --check-aux       # RGB and masks identical with and without --aux
    python vrm_render.py --sheet           # preview contact sheet with mask outlines
    python vrm_render.py --verify          # COCO loads, every instance has area > 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

import vrm_load
from render_view import camera

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "data" / "vrm_render"
W, H = 1280, 720
MAX_INST = 3
FRAMINGS = (("face", 0.55, 0.42), ("bust", 0.25, 0.75), ("full", 0.20, None))

_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import drjit as dr
    import mitsuba as mi

    class ToonIntegrator(mi.SamplingIntegrator):
        def __init__(self, props):
            mi.SamplingIntegrator.__init__(self, props)
            g = lambda k, d=0.0: float(props.get(k, d))  # noqa: E731
            L = np.array([g("light_x"), g("light_y", 1.0), g("light_z")])
            self.to_light = mi.Vector3f(*[float(x) for x in L / np.linalg.norm(L)])
            self.light = mi.Color3f(g("light_r", 1.0), g("light_g", 1.0), g("light_b", 1.0))
            self.ambient = mi.Color3f(g("amb_r", 0.3), g("amb_g", 0.3), g("amb_b", 0.3))

        def aov_names(self):
            return [f"i{k}" for k in range(MAX_INST)]

        def sample(self, scene, sampler, ray, medium=None, active=True):
            si = scene.ray_intersect(ray, active)
            hit = active & si.is_valid()
            bsdf = si.bsdf(ray)
            base = bsdf.eval_diffuse_reflectance(si, hit)
            n = si.sh_frame.n
            n = dr.select(dr.dot(n, -ray.d) < 0, -n, n)
            nl = dr.dot(n, self.to_light)
            shadow_ray = si.spawn_ray(self.to_light)
            lit = hit & (nl > 0)
            occluded = scene.ray_test(shadow_ray, lit)
            nl = dr.select(occluded, -1.0, nl)
            shape = si.shape
            toony = dr.clip(shape.eval_attribute_1("vertex_toony", si, hit), 0.0, 1.0)
            shift = shape.eval_attribute_1("vertex_shift", si, hit)
            shade = shape.eval_attribute_3("vertex_shade", si, hit)
            lo, hi = -1.0 + toony, 1.0 - toony
            t = dr.clip((nl + shift - lo) / dr.maximum(hi - lo, 1e-6), 0.0, 1.0)
            col = (base * shade + (base - base * shade) * t) * self.light + base * self.ambient
            inst = shape.eval_attribute_1("vertex_inst", si, hit)
            aovs = [dr.select(hit & (dr.abs(inst - (k + 1)) < 0.5), 1.0, 0.0)
                    for k in range(MAX_INST)]
            return dr.select(hit, col, 0.0), hit, aovs

    mi.register_integrator("vrm_toon", lambda props: ToonIntegrator(props))
    _REGISTERED = True


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


class Library:
    """Parsed avatars, their linear textures, and backgrounds, loaded once per process."""

    def __init__(self):
        import mitsuba as mi
        self.mi = mi
        self.models = read_tsv(ROOT / "vrm_models.tsv")
        self.avatars = {}
        self.tex = {}
        self.bgs = read_tsv(ROOT / "vrm_backgrounds.tsv")
        self.env = {}

    def avatar(self, name):
        if name not in self.avatars:
            self.avatars[name] = vrm_load.Avatar(ROOT / "data" / "vrm" / f"{name}.vrm")
        return self.avatars[name]

    def texture(self, key, mat):
        if key not in self.tex:
            rgb = srgb_to_linear(mat["tex"][:, :, :3].astype(np.float32) / 255.0)
            rgb = (rgb * mat["base"][None, None, :]).astype(np.float32)
            self.tex[key] = self.mi.Bitmap(rgb, pixel_format=self.mi.Bitmap.PixelFormat.RGB)
        return self.tex[key]

    def environment(self, name):
        if name not in self.env:
            bmp = self.mi.Bitmap(str(ROOT / "data" / "backgrounds" / f"{name}_4k.hdr"))
            img = np.array(bmp, dtype=np.float32)[:, :, :3]
            lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            img *= 0.18 / max(float(np.median(lum)), 1e-4)
            self.env[name] = img
        return self.env[name]


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def ray_dirs(eye, target, up, fov_deg, w=W, h=H):
    """World ray through each pixel centre, matching Mitsuba's perspective sensor with fov_axis y."""
    f, r, u = look_at(eye, target, up)
    t = math.tan(math.radians(fov_deg) / 2)
    xs = (np.arange(w) + 0.5 - w / 2) / (h / 2) * t
    ys = (h / 2 - np.arange(h) - 0.5) / (h / 2) * t
    d = f[None, None] - xs[None, :, None] * r[None, None] + ys[:, None, None] * u[None, None]
    return d / np.linalg.norm(d, axis=2, keepdims=True), f


def background(lib, rng, eye, target, up, fov, w=W, h=H):
    if rng.random() < 0.25 or not lib.bgs:
        a, b = rng.uniform(0.05, 0.95, 3), rng.uniform(0.05, 0.95, 3)
        ang = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
        s = np.clip(0.5 + (xx - 0.5) * math.cos(ang) + (yy - 0.3) * math.sin(ang), 0, 1)
        img = a[None, None] * (1 - s[..., None]) + b[None, None] * s[..., None]
        img += rng.normal(0, 0.01, img.shape)
        return srgb_to_linear(np.clip(img, 0, 1)).astype(np.float32), "gradient", img.mean((0, 1))
    name = lib.bgs[rng.integers(len(lib.bgs))]["name"]
    env = lib.environment(name)
    d, _ = ray_dirs(eye, target, up, fov, w, h)
    rot = rng.uniform(0, 2 * math.pi)
    uu = (0.5 + (np.arctan2(d[..., 0], -d[..., 2]) + rot) / (2 * math.pi)) % 1.0
    vv = np.arccos(np.clip(d[..., 1], -1, 1)) / math.pi
    eh, ew = env.shape[:2]
    x = uu * ew - 0.5
    y = np.clip(vv * eh - 0.5, 0, eh - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1, y1 = (x0 + 1) % ew, np.minimum(y0 + 1, eh - 1)
    x0 %= ew
    img = ((env[y0, x0] * (1 - fx) + env[y0, x1] * fx) * (1 - fy)
           + (env[y1, x0] * (1 - fx) + env[y1, x1] * fx) * fy)
    img = img * (1 + img / 4.0) / (1 + img)
    return img.astype(np.float32), name, env.reshape(-1, 3).mean(0)


def build_shapes(lib, rng, names):
    """Posed, scaled, placed avatars as a dict of mi.Mesh plus per-shape provenance."""
    mi = lib.mi
    shapes, meta, anchors = {}, [], []
    slots = [(0.0, 0.0)] + [(s * rng.uniform(0.55, 1.1), -rng.uniform(0.0, 0.9))
                            for s in (1, -1)]
    for k, name in enumerate(names):
        av = lib.avatar(name)
        prims, pts, height = av.pose(rng, amount=rng.uniform(0.3, 1.0))
        scale = rng.uniform(1.5, 1.7) / height
        yaw = rng.uniform(-0.45, 0.45)
        R = vrm_load.axis_angle([0, 1, 0], yaw)
        off = np.array([slots[k][0], 0.0, slots[k][1]])
        anchors.append({kk: (v * scale) @ R.T + off for kk, v in pts.items()})
        anchors[-1]["top"] = np.array([0, max(p["pos"][:, 1].max() for p in prims) * scale, 0]) + off
        for j, p in enumerate(prims):
            pos = ((p["pos"] * scale) @ R.T + off).astype(np.float32)
            faces = p["faces"]
            nrm = vrm_load.vertex_normals(pos.astype(np.float64), faces).astype(np.float32)
            mat = p["mat"]
            if mat["alpha"] != "OPAQUE":
                raise SystemExit(f"{name} primitive {j}: alphaMode {mat['alpha']} is not rendered")
            has_uv = p["uv"] is not None and mat["tex"] is not None
            m = mi.Mesh(f"a{k}_p{j}", vertex_count=len(pos), face_count=len(faces),
                        has_vertex_normals=True, has_vertex_texcoords=has_uv)
            mp = mi.traverse(m)
            mp["vertex_positions"] = mi.Float(pos.reshape(-1))
            mp["faces"] = mi.UInt(faces.reshape(-1))
            mp["vertex_normals"] = mi.Float(nrm.reshape(-1))
            if has_uv:
                mp["vertex_texcoords"] = mi.Float(p["uv"].reshape(-1))
            mp.update()
            nv = len(pos)
            m.add_attribute("vertex_inst", 1, mi.Float(np.full(nv, k + 1, np.float32)))
            m.add_attribute("vertex_toony", 1, mi.Float(np.full(nv, mat["toony"], np.float32)))
            m.add_attribute("vertex_shift", 1, mi.Float(np.full(nv, mat["shift"], np.float32)))
            m.add_attribute("vertex_shade", 3, mi.Float(np.tile(
                np.asarray(mat["shade"], np.float32), nv)))
            if has_uv:
                refl = {"type": "bitmap", "bitmap": lib.texture((name, id(mat["tex"])), mat),
                        "raw": True, "filter_type": "bilinear"}
            else:
                refl = {"type": "rgb", "value": [float(c) for c in srgb_to_linear(mat["base"])]}
            m.set_bsdf(mi.load_dict({"type": "diffuse", "reflectance": refl}))
            shapes[f"a{k}_p{j}"] = m
            meta.append({"shape": f"a{k}_p{j}", "avatar": name, "instance": k + 1,
                         "mesh": p["mesh"], "primitive": p["primitive"],
                         "has_material": mat["has_material"]})
    return shapes, meta, anchors


def pick_camera(rng, index, anchors):
    """A frontal-biased `sphere_hammersley` view of avatar 0 in one of three framings."""
    u = rng.random()
    acc = 0.0
    for framing, p, extent in FRAMINGS:
        acc += p
        if u <= acc:
            break
    a = anchors[0]
    if framing == "face":
        target = a["head"] + np.array([0, 0.09, 0])
    elif framing == "bust":
        target = (a["head"] + a["chest"]) / 2
    else:
        extent = float(a["top"][1]) * 1.05
        target = np.array([a["head"][0], a["top"][1] / 2, a["head"][2]])
    fov = rng.uniform(35, 65) if framing != "full" else rng.uniform(30, 55)
    frontal = rng.random() < 0.85
    views, offset = 4096, (rng.random(), rng.random())
    i = int(rng.integers(views))
    for _ in range(views):
        e, yaw, pitch, radius = camera(i, views, fov, offset,
                                       distance=rng.uniform(0.75, 1.25) * extent)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        ok_pitch = -0.25 < pitch < 0.5
        if ok_pitch and (not frontal or abs(yaw) < math.radians(40)):
            break
        i = (i + 1) % views
    direction = np.array([e[1], e[2], e[0]]) / np.linalg.norm(e)
    eye = target + direction * radius
    roll = rng.normal(0, 0.08)
    f = target - eye
    f /= np.linalg.norm(f)
    up = vrm_load.axis_angle(f, roll) @ np.array([0.0, 1.0, 0.0])
    return {"framing": framing, "fov": fov, "eye": eye, "target": target, "up": up,
            "yaw_deg": math.degrees(yaw), "pitch_deg": math.degrees(pitch)}


def sensor(mi, cam, spp, w=W, h=H, rfilter="gaussian", fmt="rgba"):
    return {"type": "perspective", "fov": cam["fov"], "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[float(x) for x in cam["eye"]], target=[float(x) for x in cam["target"]],
                up=[float(x) for x in cam["up"]]),
            "film": {"type": "hdrfilm", "width": w, "height": h, "rfilter": {"type": rfilter},
                     "pixel_format": fmt, "component_format": "float32"},
            "sampler": {"type": "independent", "sample_count": spp}}


def render_scene(lib, seed, spp, aux_stem=None, w=W, h=H):
    """One image: returns (uint8 RGB, list of (instance, avatar, coverage), record)."""
    import mitsuba as mi
    register()
    rng = np.random.default_rng(seed)
    count = int(rng.choice([1, 1, 2, 2, 3]))
    names = list(rng.choice([m["name"] for m in lib.models], count, replace=False))
    shapes, meta, anchors = build_shapes(lib, rng, names)
    cam = pick_camera(rng, seed, anchors)
    bg, bg_name, amb = background(lib, rng, cam["eye"], cam["target"], cam["up"], cam["fov"],
                                  w, h)
    amb = np.clip(np.asarray(amb, float), 0.02, None)
    amb = 0.18 * amb / float(amb.mean())
    L = np.array([rng.uniform(-1, 1), rng.uniform(0.3, 1.2), rng.uniform(0.2, 1.0)])
    lc = rng.uniform(0.85, 1.1, 3)
    integ = {"type": "vrm_toon"}
    for key, vec in (("light_", (L, "xyz")), ("light_", (lc, "rgb")), ("amb_", (amb, "rgb"))):
        integ.update({key + c: float(v) for c, v in zip(vec[1], vec[0])})
    scene = mi.load_dict({"type": "scene", "integrator": integ,
                          "sensor": sensor(mi, cam, spp, w, h), **shapes})
    img = np.array(mi.render(scene, spp=spp, seed=seed & 0x7FFFFFFF), dtype=np.float32)
    rgb, alpha, cov = img[..., :3], np.clip(img[..., 3:4], 0, 1), img[..., 4:4 + MAX_INST]
    out = rgb + (1 - alpha) * bg
    u8 = (linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)
    record = {"seed": int(seed), "avatars": names, "background": bg_name, "camera": {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cam.items()},
        "light_dir": L.tolist(), "spp": spp}
    if aux_stem is not None:
        record["_aux"] = write_aux(mi, scene, shapes, meta, cam, aux_stem, w, h, record)
    return u8, [(k + 1, names[k], cov[..., k]) for k in range(count)], record


def write_aux(mi, scene, shapes, meta, cam, stem, w, h, record):
    """Metric z-depth, world normals and shape_index at 1 spp from pixel centres; own scene."""
    aux_scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "pp:position,nn:sh_normal,si:shape_index"},
        "sensor": sensor(mi, cam, 1, w, h, rfilter="box"), **shapes})
    a = np.array(mi.render(aux_scene, spp=1, seed=0), dtype=np.float32)
    if a.shape[2] != 7:
        raise SystemExit(f"aux pass produced {a.shape[2]} channels, expected position+normal+index")
    # z from the hit position: Mitsuba's depth AOV is measured from the near-clip plane.
    pos, normal, sidx = a[..., 0:3], a[..., 3:6], a[..., 6]
    f = look_at(np.asarray(cam["eye"]), np.asarray(cam["target"]), np.asarray(cam["up"]))[0]
    hit = np.abs(pos).sum(-1) > 0
    zdepth = np.where(hit, (pos - np.asarray(cam["eye"])) @ f, 0.0).astype(np.float32)
    mi.Bitmap(np.ascontiguousarray(zdepth[..., None]), pixel_format=mi.Bitmap.PixelFormat.Y
              ).write(str(stem) + ".depth.exr")
    mi.Bitmap(np.ascontiguousarray(normal.astype(np.float16)),
              pixel_format=mi.Bitmap.PixelFormat.RGB).write(str(stem) + ".normal.exr")
    order = [s.id() for s in scene.shapes()]
    by_id = {m["shape"]: m for m in meta}
    fy = (h / 2) / math.tan(math.radians(cam["fov"]) / 2)
    f_, r_, u_ = look_at(np.asarray(cam["eye"]), np.asarray(cam["target"]),
                         np.asarray(cam["up"]))
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = -r_, u_, f_, cam["eye"]
    json.dump({"width": w, "height": h, "fov_y_deg": cam["fov"],
               "intrinsics": {"fx": fy, "fy": fy, "cx": w / 2, "cy": h / 2},
               "depth": "z-depth along the optical axis, metres; 0 where no surface",
               "normal": "world-space shading normal, Y up, float16",
               "camera_to_world": c2w.tolist(), "camera_axes": "x left, y up, z forward",
               "shape_index": {str(i): by_id[s] for i, s in enumerate(order)},
               "shape_index_file": "<stem>.shape_index.npy, int16, -1 = background",
               "seed": record["seed"]},
              open(str(stem) + ".aux.json", "w"), indent=1)
    np.save(str(stem) + ".shape_index.npy", np.where(hit, sidx, -1).astype(np.int16))
    return pos, hit


def masks_to_coco(inst, image_id, ann_id):
    from pycocotools import mask as mask_util
    anns = []
    for k, name, cov in inst:
        m = np.asfortranarray((cov >= 0.5).astype(np.uint8))
        area = int(m.sum())
        if area < 200:
            continue
        rle = mask_util.encode(m)
        x, y, bw, bh = mask_util.toBbox(rle).tolist()
        rle["counts"] = rle["counts"].decode("ascii")
        anns.append({"id": ann_id + len(anns), "image_id": image_id, "category_id": 1,
                     "segmentation": rle, "area": area, "bbox": [x, y, bw, bh],
                     "iscrowd": 0, "avatar": name, "instance": k})
    return anns


def setup(threads):
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    if threads:
        dr.set_thread_count(threads)


def run(count, spp, aux, start):
    lib = Library()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    if aux:
        (OUT / "aux").mkdir(parents=True, exist_ok=True)
    images, anns, times = [], [], []
    for n in range(start, start + count):
        t0 = time.time()
        fname = f"vrm_{n:06d}.png"
        stem = OUT / "aux" / f"vrm_{n:06d}" if aux else None
        u8, inst, rec = render_scene(lib, 1000003 * (n + 1), spp, stem)
        rec.pop("_aux", None)
        Image.fromarray(u8).save(OUT / "images" / fname)
        a = masks_to_coco(inst, n + 1, len(anns) + 1)
        anns += a
        images.append({"id": n + 1, "file_name": fname, "width": W, "height": H, **rec})
        times.append(time.time() - t0)
        print(f"{fname} {rec['camera']['framing']:4s} {len(a)} inst "
              f"{','.join(rec['avatars'])} {times[-1]:.1f}s", flush=True)
    coco = {"info": {"description": "Posed VRM avatar renders, Mitsuba 3 toon integrator",
                     "licences": "vrm_models.tsv, vrm_backgrounds.tsv"},
            "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    json.dump(coco, open(OUT / "annotations.json", "w"))
    print(f"{len(images)} images, {len(anns)} instances, "
          f"median {np.median(times):.1f}s/image")


def unprojection_error_mm(stem, pos, hit, z_offset=0.0):
    """Median distance between the hit and the depth pixel unprojected through aux.json."""
    import mitsuba as mi
    j = json.load(open(str(stem) + ".aux.json"))
    z = np.array(mi.Bitmap(str(stem) + ".depth.exr")).reshape(j["height"], j["width"]) + z_offset
    K, c2w = j["intrinsics"], np.array(j["camera_to_world"])
    v, u = np.mgrid[0:j["height"], 0:j["width"]] + 0.5
    pc = np.stack([-(u - K["cx"]) / K["fx"] * z, -(v - K["cy"]) / K["fy"] * z, z], -1)
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
    return float(np.median(np.linalg.norm(pw[hit] - pos[hit], axis=1))) * 1000


def check_aux(tmp):
    """The aux pass must leave RGB and masks byte-identical."""
    lib = Library()
    tmp.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in (7, 11):
        a_rgb, a_inst, _ = render_scene(lib, seed, 4, None, 320, 180)
        b_rgb, b_inst, rec = render_scene(lib, seed, 4, tmp / f"chk{seed}", 320, 180)
        results.append((f"seed {seed}: unprojected depth lands within 2 mm of the hit",
                        unprojection_error_mm(tmp / f"chk{seed}", *rec["_aux"]) < 2.0))
        results.append((f"seed {seed}: control: a near-clip-sized 10 mm depth offset is caught",
                        unprojection_error_mm(tmp / f"chk{seed}", *rec["_aux"], 0.01) >= 2.0))
        same_rgb = a_rgb.tobytes() == b_rgb.tobytes()
        same_mask = all(np.array_equal(x[2] >= 0.5, y[2] >= 0.5) for x, y in zip(a_inst, b_inst))
        results.append((f"seed {seed}: RGB identical with and without aux", same_rgb))
        results.append((f"seed {seed}: masks identical with and without aux", same_mask))
        results.append((f"seed {seed}: aux files written",
                        (tmp / f"chk{seed}.depth.exr").exists()))
    c_rgb, _, _ = render_scene(lib, 11, 5, None, 320, 180)
    results.append(("control: one extra sample per pixel is detected",
                    c_rgb.tobytes() != a_rgb.tobytes()))
    for name, ok in results:
        print(("  ok   " if ok else "  FAIL ") + name)
    return 0 if all(ok for _, ok in results) else 1


def verify_coco(coco):
    """COCO loads in pycocotools; every instance has area > 0 and matches its decoded RLE."""
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO
    api = COCO()
    api.dataset = coco
    api.createIndex()
    bad = [a["id"] for a in coco["annotations"]
           if a["area"] <= 0 or int(mask_util.area(a["segmentation"])) != a["area"]
           or a["category_id"] != 1 or a["iscrowd"] != 0]
    return len(api.getImgIds()), len(api.getAnnIds()), bad


def verify():
    coco = json.load(open(OUT / "annotations.json"))
    n_img, n_ann, bad = verify_coco(coco)
    missing = [i["file_name"] for i in coco["images"] if not (OUT / "images" / i["file_name"]).exists()]
    empty = sum(1 for i in coco["images"] if not any(a["image_id"] == i["id"] for a in coco["annotations"]))
    broken = json.loads(json.dumps(coco))
    broken["annotations"][0]["area"] = 0
    control = len(verify_coco(broken)[2]) > 0
    print(f"  {n_img} images, {n_ann} instances, {len(bad)} bad instances, "
          f"{len(missing)} missing files, {empty} images with no instance")
    print(("  ok   " if control else "  FAIL ") + "control: a zero-area instance is caught")
    return 0 if not bad and not missing and control else 1


def sheet(n=20, cols=5, path=None):
    from pycocotools import mask as mask_util
    coco = json.load(open(OUT / "annotations.json"))
    imgs = coco["images"]
    step = max(1, len(imgs) // n)
    pick = imgs[::step][:n]
    tw, th = 384, 216
    canvas = Image.new("RGB", (cols * tw, ((len(pick) + cols - 1) // cols) * th), "black")
    colours = [(255, 60, 60), (60, 255, 60), (60, 160, 255)]
    for i, im in enumerate(pick):
        base = np.asarray(Image.open(OUT / "images" / im["file_name"]).convert("RGB")).copy()
        for a in coco["annotations"]:
            if a["image_id"] != im["id"]:
                continue
            m = mask_util.decode(a["segmentation"]).astype(bool)
            edge = m & ~(np.roll(m, 2, 0) & np.roll(m, -2, 0) & np.roll(m, 2, 1) & np.roll(m, -2, 1))
            base[edge] = colours[(a["instance"] - 1) % 3]
        tile = Image.fromarray(base).resize((tw, th))
        ImageDraw.Draw(tile).text((4, 4), f"{im['file_name']} {im['camera']['framing']}",
                                  fill=(255, 255, 0))
        canvas.paste(tile, ((i % cols) * tw, (i // cols) * th))
    canvas.save(path or OUT / "preview.png")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--check-aux", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.sheet:
        sheet()
        return 0
    if args.verify:
        return verify()
    setup(args.threads)
    if args.check_aux:
        return check_aux(OUT / "check_aux")
    run(args.count, args.spp, args.aux, args.start)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
