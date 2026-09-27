# SPDX-License-Identifier: Apache-2.0 OR MIT
"""A VRM (0.x or 1.0) as numpy primitives, posed by linear blend skinning.

A VRM is a GLB; skins, humanoid bone map and materials come from standard glTF plus the
VRM/VRMC_vrm extensions. Output is Y-up, metres, the avatar facing +Z with feet at y=0.

    python vrm_load.py <file.vrm> [--self-test]
"""
from __future__ import annotations

import io
import pathlib
import sys

import numpy as np
import pygltflib
from PIL import Image

import mtoon

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NEUTRAL_BASE = (0.55, 0.55, 0.55)
TOONY = mtoon.DEFAULTS["shading_toony_factor"]
SHIFT = mtoon.DEFAULTS["shading_shift_factor"]


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class Avatar:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.name = self.path.stem
        g = pygltflib.GLTF2().load_binary(str(path))
        self.g, self.blob = g, g.binary_blob()
        self._local()
        self.parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
        self.bones = self._humanoid()
        self.images = {}
        self.prims = self._primitives()

    def accessor(self, i):
        a = self.g.accessors[i]
        bv = self.g.bufferViews[a.bufferView]
        dt, w = np.dtype(COMPONENT[a.componentType]), WIDTH[a.type]
        start = (bv.byteOffset or 0) + (a.byteOffset or 0)
        stride = bv.byteStride or dt.itemsize * w
        raw = np.frombuffer(self.blob, np.uint8, count=stride * (a.count - 1) + dt.itemsize * w,
                            offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, (a.count, dt.itemsize * w), (stride, 1))
        out = np.ascontiguousarray(rows).view(dt).reshape(a.count, w)
        if a.normalized:
            out = out.astype(np.float32) / np.iinfo(dt).max
        return out

    def _local(self):
        self.rest_local = []
        for n in self.g.nodes:
            m = np.eye(4)
            if n.matrix:
                m = np.array(n.matrix, float).reshape(4, 4).T
            else:
                t, r, s = n.translation or [0, 0, 0], n.rotation or [0, 0, 0, 1], n.scale or [1, 1, 1]
                m[:3, :3] = quat_to_mat(r) @ np.diag(s)
                m[:3, 3] = t
            self.rest_local.append(m)

    def globals(self, local):
        out = [None] * len(local)

        def go(i):
            if out[i] is None:
                p = self.parent.get(i)
                out[i] = local[i] if p is None else go(p) @ local[i]
            return out[i]
        for i in range(len(local)):
            go(i)
        return out

    def _humanoid(self):
        ext = self.g.extensions or {}
        if "VRMC_vrm" in ext:
            hb = ext["VRMC_vrm"]["humanoid"]["humanBones"]
            return {k: v["node"] for k, v in hb.items()}
        if "VRM" in ext:
            return {b["bone"]: b["node"] for b in ext["VRM"]["humanoid"]["humanBones"]}
        return {}

    def image(self, idx):
        if idx not in self.images:
            im = self.g.images[idx]
            bv = self.g.bufferViews[im.bufferView]
            data = self.blob[bv.byteOffset or 0:(bv.byteOffset or 0) + bv.byteLength]
            self.images[idx] = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        return self.images[idx]

    def material(self, mi_):
        """(base texture RGBA or None, base factor, shade factor, toony, shift, alpha mode)."""
        if mi_ is None:
            return {"tex": None, "base": np.array(NEUTRAL_BASE), "shade": 0.6,
                    "toony": TOONY, "shift": SHIFT, "has_material": False,
                    "alpha": "OPAQUE"}
        m = self.g.materials[mi_]
        pbr = m.pbrMetallicRoughness
        base = np.array((pbr.baseColorFactor if pbr else None) or [1, 1, 1, 1], float)[:3]
        tex = None
        if pbr and pbr.baseColorTexture is not None:
            tex = self.image(self.g.textures[pbr.baseColorTexture.index].source)
        shade, toony, shift = None, TOONY, SHIFT
        ext = self.g.extensions or {}
        mext = m.extensions or {}
        if "VRMC_materials_mtoon" in mext:
            mt = mext["VRMC_materials_mtoon"]
            shade = np.array(mt.get("shadeColorFactor", [0.6] * 3), float)
            toony = mt.get("shadingToonyFactor", TOONY)
            shift = mt.get("shadingShiftFactor", SHIFT)
        elif "VRM" in ext:
            props = ext["VRM"].get("materialProperties", [])
            p = next((x for x in props if x.get("name") == m.name), None)
            if p and p.get("shader") == "VRM/MToon":
                vec, fl = p.get("vectorProperties", {}), p.get("floatProperties", {})
                col = np.array(vec.get("_Color", [1, 1, 1, 1])[:3], float)
                sh = np.array(vec.get("_ShadeColor", [0.6, 0.6, 0.6, 1])[:3], float)
                shade = sh / np.maximum(col, 1e-3)
                toony = fl.get("_ShadeToony", TOONY)
                shift = fl.get("_ShadeShift", SHIFT)
        if shade is None:
            shade = np.full(3, 0.6)
        return {"tex": tex, "base": base, "shade": np.clip(shade, 0, 1), "toony": toony,
                "shift": shift, "has_material": True, "alpha": m.alphaMode or "OPAQUE"}

    def _primitives(self):
        out = []
        for ni, n in enumerate(self.g.nodes):
            if n.mesh is None:
                continue
            skin = self.g.skins[n.skin] if n.skin is not None else None
            for pi, p in enumerate(self.g.meshes[n.mesh].primitives):
                if p.mode not in (None, 4):
                    continue
                at = p.attributes
                pos = self.accessor(at.POSITION).astype(np.float64)
                idx = (self.accessor(p.indices).reshape(-1) if p.indices is not None
                       else np.arange(len(pos)))
                prim = {"node": ni, "mesh": n.mesh, "primitive": pi,
                        "pos": pos, "faces": idx.reshape(-1, 3).astype(np.uint32),
                        "uv": (self.accessor(at.TEXCOORD_0).astype(np.float32)
                               if at.TEXCOORD_0 is not None else None),
                        "mat": self.material(p.material)}
                if skin is not None and at.JOINTS_0 is not None:
                    prim["joints"] = self.accessor(at.JOINTS_0).astype(np.int64)
                    w = self.accessor(at.WEIGHTS_0).astype(np.float64)
                    prim["weights"] = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
                    prim["skin_joints"] = list(skin.joints)
                    prim["ibm"] = (self.accessor(skin.inverseBindMatrices)
                                   .reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
                                   if skin.inverseBindMatrices is not None
                                   else np.tile(np.eye(4), (len(skin.joints), 1, 1)))
                out.append(prim)
        return out

    def frame(self, G):
        """Character left, up and forward from the rest skeleton; +X is left in glTF."""
        up = np.array([0.0, 1.0, 0.0])
        b = self.bones
        if "leftUpperArm" in b and "rightUpperArm" in b:
            left = G[b["leftUpperArm"]][:3, 3] - G[b["rightUpperArm"]][:3, 3]
            left[1] = 0
            left /= np.linalg.norm(left)
        else:
            left = np.array([1.0, 0.0, 0.0])
        return left, up, np.cross(left, up)

    def pose(self, rng=None, amount=1.0):
        """World-space posed primitives plus the head position, normalised to +Z forward."""
        G0 = self.globals(self.rest_local)
        left, up, fwd = self.frame(G0)
        deltas = {}
        if rng is not None:
            def r(a, b):
                return rng.uniform(a, b) * amount
            for side, sgn in (("left", -1.0), ("right", 1.0)):
                drop = rng.uniform(0.9, 1.3) if rng.random() < 0.8 else rng.uniform(-0.2, 0.4)
                deltas[side + "UpperArm"] = (axis_angle(up, sgn * r(-0.3, 0.5))
                                             @ axis_angle(fwd, sgn * drop))
                deltas[side + "LowerArm"] = axis_angle(up, sgn * r(0.0, 1.2))
                deltas[side + "UpperLeg"] = axis_angle(left, r(-0.25, 0.15))
            deltas["head"] = (axis_angle(up, r(-0.5, 0.5)) @ axis_angle(left, r(-0.25, 0.3))
                              @ axis_angle(fwd, r(-0.2, 0.2)))
            deltas["neck"] = axis_angle(up, r(-0.2, 0.2))
            deltas["spine"] = axis_angle(up, r(-0.2, 0.2)) @ axis_angle(left, r(-0.1, 0.15))
        local = list(self.rest_local)
        for bone, Q in deltas.items():
            ni = self.bones.get(bone)
            if ni is None:
                continue
            R = G0[ni][:3, :3]
            D = np.eye(4)
            D[:3, :3] = np.linalg.inv(R) @ Q @ R
            local[ni] = local[ni] @ D
        G = self.globals(local)
        prims = []
        for p in self.prims:
            if "joints" in p:
                J = np.stack([G[j] @ p["ibm"][k] for k, j in enumerate(p["skin_joints"])])
                M = np.einsum("vk,vkij->vij", p["weights"], J[p["joints"]])
                pos = np.einsum("vij,vj->vi", M[:, :3, :3], p["pos"]) + M[:, :3, 3]
            else:
                M = G[p["node"]]
                pos = p["pos"] @ M[:3, :3].T + M[:3, 3]
            prims.append(dict(p, pos=pos))
        head = G[self.bones["head"]][:3, 3] if "head" in self.bones else None
        chest = G[self.bones.get("upperChest", self.bones.get("chest", self.bones.get(
            "spine", 0)))][:3, 3]
        yaw = np.arctan2(fwd[0], fwd[2])
        Ry = axis_angle(up, -yaw)
        allp = np.concatenate([q["pos"] for q in prims]) @ Ry.T
        floor = allp[:, 1].min()
        centre = np.array([(allp[:, 0].min() + allp[:, 0].max()) / 2, floor,
                           (allp[:, 2].min() + allp[:, 2].max()) / 2])
        for q in prims:
            q["pos"] = q["pos"] @ Ry.T - centre
        pts = {"head": head, "chest": chest}
        pts = {k: (v @ Ry.T - centre) for k, v in pts.items() if v is not None}
        return prims, pts, float(allp[:, 1].max() - floor)


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    tri = verts[faces.astype(np.int64)]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, faces[:, i].astype(np.int64), fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n[ln[:, 0] == 0] = (0.0, 1.0, 0.0)
    ln[ln[:, 0] == 0] = 1.0
    return n / ln


def self_test(path):
    a = Avatar(path)
    prims, pts, h = a.pose()
    ok = [("has primitives", len(prims) > 0),
          ("has a head bone", "head" in pts),
          ("height is human-scale (0.5-2.5 m)", 0.5 < h < 2.5),
          ("head above chest", pts["head"][1] > pts["chest"][1]),
          ("feet on the floor", abs(min(p["pos"][:, 1].min() for p in prims)) < 1e-6)]
    rng = np.random.default_rng(0)
    posed, _, _ = a.pose(rng)
    ok.append(("a random pose moves vertices",
               any(np.abs(p["pos"] - q["pos"]).max() > 0.01 for p, q in zip(prims, posed))))
    flipped, _, _ = a.pose()
    ok.append(("the rest pose is deterministic",
               all(np.array_equal(p["pos"], q["pos"]) for p, q in zip(prims, flipped))))
    for name, good in ok:
        print(("  ok   " if good else "  FAIL ") + name)
    print(f"  height {h:.2f} m, {sum(len(p['faces']) for p in prims)} triangles")
    return 0 if all(g for _, g in ok) else 1


if __name__ == "__main__":
    sys.exit(self_test(sys.argv[1]))
