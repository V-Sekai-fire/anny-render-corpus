# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Fetch CC0/CC-BY VRM avatars into data/vrm/ and write vrm_models.tsv.

A model is kept only when its collection licence and the licence in its own VRM meta both
read CC0 or CC-BY. The sha256 of each kept file is pinned in the TSV; a later run verifies it.

    python vrm_fetch.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
VRM_DIR = ROOT / "data" / "vrm"
TSV = ROOT / "vrm_models.tsv"
REGISTRY_REV = "0f9a1b2fd99894736563d55b2c9dc9125700d081"
REGISTRY = ("https://raw.githubusercontent.com/toxsam/open-source-avatars/"
            + REGISTRY_REV + "/data/")
REGISTRY_PICKS = {
    "100avatars-r1": ["Erika", "Kate", "Kyle", "Lydia", "Olivia", "Rose", "Samuela", "Shiro",
                      "Jennifer", "Robert", "Mikel", "Witch", "Ro"],
    "100avatars-r2": ["MoonGirl"],
    "100avatars-r3": ["Anna", "CursedAmy", "Eugenia", "Juanita", "LadyFawn", "StitchWitch",
                      "AlienTeen", "Jenny", "Vladi"],
    "xmas-chibis": ["Elel Silverbell", "Holy Garland", "Comet Kringle", "Frosty Claus",
                    "Pip Poinsenttia", "Crystal Yuletide", "Clara Nutcracker", "Nick North"],
    "halloween-rising": ["Mocking Spit: Strawberry", "Esktix: Midnight", "Harvester: Wheat"],
    "toxsam": ["Orion", "Aurora"],
}
GATEWAYS = ("https://dweb.link/ipfs/", "https://ipfs.io/ipfs/")
PINNED_GATEWAY = "https://gateway.pinata.cloud/ipfs/"
MAKERS = {"100avatars-r1": "Polygonal Mind", "100avatars-r2": "Polygonal Mind",
          "100avatars-r3": "Polygonal Mind", "xmas-chibis": "VIPE",
          "halloween-rising": "VIPE", "toxsam": "ToxSam"}
DIRECT = [
    ("AvatarSample_B", "https://raw.githubusercontent.com/pixiv/ChatVRM/"
     "main/public/AvatarSample_B.vrm", "pixiv Inc."),
    ("Seed-san", "https://raw.githubusercontent.com/vrm-c/vrm-specification/"
     "master/samples/Seed-san/vrm/Seed-san.vrm", "VRM Consortium"),
    ("VRM1_Constraint_Twist_Sample", "https://raw.githubusercontent.com/pixiv/three-vrm/"
     "dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm", "pixiv Inc."),
]
FIELDS = ["name", "licence", "url", "attribution", "source", "meta_licence", "sha256",
          "prim_has_material"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "anny-render-corpus/vrm_fetch"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def glb_json(blob):
    if blob[:4] != b"glTF":
        raise ValueError("not a GLB")
    n = int.from_bytes(blob[12:16], "little")
    return json.loads(blob[20:20 + n])


def meta_licence(gltf):
    """The licence the file states about itself, normalised to CC0 / CC-BY / other text."""
    ext = gltf.get("extensions", {})
    if "VRMC_vrm" in ext:
        m = ext["VRMC_vrm"].get("meta", {})
        url = m.get("licenseUrl", "")
        other = m.get("otherLicenseUrl", "")
        text = f"{url} {other}".strip()
        if "creativecommons.org/publicdomain/zero" in text:
            return "CC0"
        if "creativecommons.org/licenses/by/" in text:
            return "CC-BY"
        return "VRM-1.0-licence:" + text
    if "VRM" in ext:
        m = ext["VRM"].get("meta", {})
        name = m.get("licenseName", "")
        return {"CC0": "CC0", "CC_BY": "CC-BY"}.get(name, name + ":" + m.get("otherLicenseUrl", ""))
    return "none"


def prim_materials(gltf):
    return ",".join("yes" if "material" in p else "no"
                    for m in gltf.get("meshes", []) for p in m["primitives"])


def open_ok(lic):
    return lic in ("CC0", "CC-BY")


def candidates():
    projects = {p["id"]: p for p in json.loads(get(REGISTRY + "projects.json"))}
    for pid, names in REGISTRY_PICKS.items():
        lic = projects[pid]["license"]
        rows = {a["name"]: a for a in json.loads(get(REGISTRY + f"avatars/{pid}.json"))}
        for n in names:
            a = rows[n]
            url = a["model_file_url"]
            for g in GATEWAYS:
                url = url.replace(g, PINNED_GATEWAY)
            slug = "".join(ch if ch.isalnum() else "_" for ch in n).strip("_")
            yield {"name": f"{pid}_{slug}", "licence": lic, "url": url,
                   "attribution": f"{n}, {projects[pid]['name']} by {MAKERS[pid]}, "
                                  "via Open Source Avatars registry",
                   "source": REGISTRY + f"avatars/{pid}.json"}
    for n, url, who in DIRECT:
        yield {"name": n, "licence": "meta", "url": url, "attribution": who, "source": url}


def load_tsv():
    if not TSV.exists():
        return {}
    with TSV.open() as f:
        return {r["name"]: r for r in csv.DictReader(f, delimiter="\t")}


def write_tsv(rows):
    with TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["name"]):
            w.writerow({k: r[k] for k in FIELDS})


HDRIS = ["abandoned_bakery", "art_studio", "bathroom", "billiard_hall", "blue_photo_studio",
         "brown_photostudio_02", "ballroom", "aft_lounge", "anniversary_lounge", "boiler_room",
         "artist_workshop", "blinds", "autoshop_01", "bank_vault", "kiara_interior",
         "lebombo", "hotel_room", "venice_sunset", "kloppenheim_06", "rooftop_night",
         "shanghai_bund", "park_music_stage", "studio_small_09", "neon_photostudio"]
BG_DIR = ROOT / "data" / "backgrounds"
BG_TSV = ROOT / "vrm_backgrounds.tsv"
BG_FIELDS = ["name", "licence", "url", "attribution", "sha256"]


def fetch_backgrounds():
    BG_DIR.mkdir(parents=True, exist_ok=True)
    pinned = {}
    if BG_TSV.exists():
        with BG_TSV.open() as f:
            pinned = {r["name"]: r for r in csv.DictReader(f, delimiter="\t")}
    rows = []
    for n in HDRIS:
        p = BG_DIR / f"{n}_4k.hdr"
        try:
            files = json.loads(get(f"https://api.polyhaven.com/files/{n}"))
            info = json.loads(get(f"https://api.polyhaven.com/info/{n}"))
        except Exception as e:
            print(f"skip background {n}: {e}")
            continue
        url = files["hdri"]["4k"]["hdr"]["url"]
        if not p.exists():
            p.write_bytes(get(url))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if n in pinned and pinned[n]["sha256"] != digest:
            print(f"skip background {n}: sha256 differs from pinned")
            continue
        rows.append({"name": n, "licence": "CC0", "url": url,
                     "attribution": ", ".join(info["authors"]) + " / Poly Haven",
                     "sha256": digest})
        print(f"background {n}")
    with BG_TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, BG_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify pinned hashes only")
    args = ap.parse_args(argv)
    VRM_DIR.mkdir(parents=True, exist_ok=True)
    pinned = load_tsv()
    if args.check:
        bad = 0
        for n, r in pinned.items():
            p = VRM_DIR / f"{n}.vrm"
            ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
            bad += not ok
            print(("ok  " if ok else "FAIL"), n)
        return 1 if bad else 0

    kept, skipped, total = [], [], 0
    for c in candidates():
        p = VRM_DIR / f"{c['name']}.vrm"
        if not p.exists():
            try:
                p.write_bytes(get(c["url"]))
            except Exception as e:
                skipped.append((c["name"], f"download: {e}"))
                continue
        blob = p.read_bytes()
        total += len(blob)
        digest = hashlib.sha256(blob).hexdigest()
        if c["name"] in pinned and pinned[c["name"]]["sha256"] != digest:
            skipped.append((c["name"], "sha256 differs from pinned"))
            continue
        gltf = glb_json(blob)
        ml = meta_licence(gltf)
        coll = ml if c["licence"] == "meta" else c["licence"]
        if not (open_ok(coll) and open_ok(ml)):
            skipped.append((c["name"], f"licence collection={coll} meta={ml}"))
            p.unlink()
            continue
        c.update(licence=coll, meta_licence=ml, sha256=digest,
                 prim_has_material=prim_materials(gltf))
        kept.append(c)
        print(f"kept {c['name']:40s} {coll:6s} {len(blob) / 1e6:6.1f} MB")
    for n, why in skipped:
        print(f"skip {n}: {why}")
    write_tsv(kept)
    fetch_backgrounds()
    print(f"{len(kept)} kept, {len(skipped)} skipped, {total / 1e6:.0f} MB on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
