"""Stage A of sinew-sim dev.3: dump ANNY ground-truth COCO-17 3D joints.

The single-pose render corpus under `anny-render-corpus-constructed/renders/` is
"one identity, one subject, one pose". To let a separate numpy stage compute
absolute MPJPE, we need the ground-truth 3D joints of exactly the pose that was
rendered, in ANNY world metres, plus the normalisation the render applied.

WHICH MESH IS A MEASUREMENT, NOT AN ASSUMPTION. Each render sidecar records the
(centre, scale) that `render_view.normalise()` produced from the rendered vertices.
We rebuild candidate meshes deterministically, recompute (centre, scale) from each
candidate's verts via the SAME `normalise()`, and the candidate whose deltas vanish
to machine precision is the rendered mesh. If none match within a tight tolerance we
HARD STOP rather than emit GT for an unverified pose.

CANDIDATES. The dev.3 brief expected the rest/rank1/rank5 bootstrap meshes from
`generate_bootstrap_poses.py` (built on the `soma` rig/topology). Measured, none of
those three match: the rendered corpus was produced by `render_grid.py` from the
MakeHuman basemesh topology that `render_corpus.py` pins --
`Anny(topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False))`,
19,158 vertices, the topology `anny/data/keypoints/coco.pth` is indexed against --
at the identity (rest) pose. So we test that candidate too, and it matches to ~2e-16.
We regress the joints from whichever candidate's model the match belongs to, so the
regression topology is the rendered topology.

Reuses:
  - render_view.normalise          -- the exact centre/scale recipe the render used
  - generate_bootstrap_poses logic -- the exact soma posed-mesh construction
  - render_corpus.load_model logic  -- the exact makehuman-basemesh construction
  - KeypointsRegressor.coco        -- the exact 3D keypoint regression (W @ V)

Usage:
    pixi run --environment anny python dump_gt_joints_dev3.py [out.json]
"""

import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch

import anny
from anny import Anny
from anny.keypoints import KeypointsRegressor
from anny.models.model_data import TopologyConfig

# Reuse the render's exact normalisation recipe -- do not reinvent it.
from render_view import normalise

# COCO-17 keypoint names, in the canonical COCO order. The regressor's labels are
# COCO names (identity-by-name), so passing these as `labels` selects exactly these
# 17 joints in this order -- no mapping table needed.
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Same constants generate_bootstrap_poses.py uses.
PERTURBATION_SIGMA_RAD = 0.05
DEFAULT_SEED = 0
BASEMESH_VERTS = 19158

# Resolve paths without hard-coding machine-specific locations (house rule: no-local-paths).
# The constructed corpus is a sibling of this repo dir; override with ANNY_RENDERS_DIR.
_REPO_DIR = pathlib.Path(__file__).resolve().parent
CONSTRUCTED_RENDERS = pathlib.Path(
    os.environ.get(
        "ANNY_RENDERS_DIR",
        str(_REPO_DIR.parent / "anny-render-corpus-constructed" / "renders"),
    )
)
# Default output next to the script; the caller passes an explicit path (argv[1] or
# DEV3_GT_JSON) to write elsewhere (e.g. a session scratchpad). Never ship a session path.
DEFAULT_OUT = pathlib.Path(
    os.environ.get("DEV3_GT_JSON", str(_REPO_DIR / "gt_joints_dev3.json"))
)

MATCH_TOL = 1e-4  # tight: a true match sits at machine precision, well under this.


def build_soma_model():
    """The soma rig/topology model generate_bootstrap_poses.py builds."""
    model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref")
    model.eval()
    return model


def build_basemesh_model():
    """The MakeHuman basemesh model render_corpus.py pins (coco.pth's topology)."""
    model = Anny(
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False)
    )
    model.eval()
    out = model()
    n = out["vertices"].shape[1]
    if n != BASEMESH_VERTS:
        raise SystemExit(
            f"basemesh topology returns {n} vertices, expected {BASEMESH_VERTS}"
        )
    return model


def soma_candidates(model, seed=DEFAULT_SEED, sigma_rad=PERTURBATION_SIGMA_RAD):
    """Rebuild rest / rank1 / rank5 exactly as generate_bootstrap_poses.py.

    Returns {pose_id: verts (N,3) float64}. rank1 is deliberately identical to rest.
    """
    n_bones = len(model.bone_labels)
    if n_bones != 78:
        raise SystemExit(
            f"expected 78 bones for soma rig, got {n_bones}: {model.bone_labels[:5]}..."
        )

    with torch.no_grad():
        rest = model()
    verts_rest = rest["rest_vertices"][0].numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    pose_rank5 = rng.normal(0.0, sigma_rad, size=(n_bones, 3)).astype(np.float64)

    import roma
    rotvec = torch.from_numpy(pose_rank5[None])              # [1, 78, 3] float64
    R = roma.rotvec_to_rotmat(rotvec)                        # [1, 78, 3, 3]
    T = torch.zeros(R.shape[0], R.shape[1], 4, 4, dtype=torch.float64)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    with torch.no_grad():
        posed = model(pose_parameters=T)
    verts_rank5 = posed["vertices"][0].numpy().astype(np.float64)

    return {
        "rest": verts_rest,
        "rank1": verts_rest.copy(),   # exactly rest, by construction
        "rank5": verts_rank5,
    }


def regress_joints(model, verts):
    """COCO-17 world-metre joints from a posed mesh via the coco.pth regressor."""
    kp = KeypointsRegressor.coco(model, labels=COCO17)
    if list(kp.labels) != COCO17:
        raise SystemExit(
            f"regressor labels differ from COCO17 request:\n  got {list(kp.labels)}\n"
            f"  want {COCO17}"
        )
    with torch.no_grad():
        joints = kp(
            {"vertices": torch.from_numpy(np.asarray(verts, dtype=np.float64)[None])}
        )[0].numpy().astype(np.float64)
    if joints.shape != (17, 3):
        raise SystemExit(f"expected (17,3) joints, got {joints.shape}")
    return joints


def main(argv):
    out_path = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT

    # One sidecar supplies the normalisation the render pinned. All frames in this
    # single-pose corpus share the same mesh, hence the same (centre, scale).
    sidecars = sorted(glob.glob(str(CONSTRUCTED_RENDERS / "*.json")))
    if not sidecars:
        raise SystemExit(f"no sidecar JSON under {CONSTRUCTED_RENDERS}")
    sidecar = json.loads(pathlib.Path(sidecars[0]).read_text())
    sidecar_centre = np.array(sidecar["normalisation"]["centre"], dtype=np.float64)
    sidecar_scale = float(sidecar["normalisation"]["scale"])

    # Build both models and the full candidate set. Each candidate carries the model
    # whose topology produced it, so regression uses the matched mesh's own topology.
    soma_model = build_soma_model()
    basemesh_model = build_basemesh_model()

    candidates = {}  # id -> (model, verts, description)
    for pid, verts in soma_candidates(soma_model).items():
        candidates[pid] = (
            soma_model, verts,
            f"generate_bootstrap_poses.py soma {pid}",
        )
    candidates["rest_basemesh"] = (
        basemesh_model, basemesh_model()["vertices"][0].numpy().astype(np.float64),
        "render_corpus.py MakeHuman basemesh (keep-unattached), rest/identity pose",
    )

    # POSE-PIN: recompute (centre, scale) from each candidate via the render's own
    # normalise(), compare to the sidecar.
    deltas = {}
    for pid, (_, verts, _desc) in candidates.items():
        _, c, s = normalise(verts)
        deltas[pid] = {
            "recomputed_centre": [float(x) for x in c],
            "recomputed_scale": float(s),
            "delta_centre": float(np.linalg.norm(c - sidecar_centre)),
            "delta_scale": float(abs(s - sidecar_scale)),
            "n_vertices": int(verts.shape[0]),
        }

    print("POSE-PIN candidate deltas (vs sidecar "
          f"centre={sidecar_centre.tolist()} scale={sidecar_scale}):")
    for pid, d in deltas.items():
        print(f"  {pid:14s}  nV={d['n_vertices']:6d}  "
              f"delta_centre={d['delta_centre']:.3e}  delta_scale={d['delta_scale']:.3e}")

    matches = [p for p, d in deltas.items()
               if d["delta_centre"] <= MATCH_TOL and d["delta_scale"] <= MATCH_TOL]

    if not matches:
        raise SystemExit(
            "POSE-PIN FAILED: no candidate matched the sidecar normalisation within "
            f"tol={MATCH_TOL}. Refusing to emit GT for an unverified pose.\n"
            f"  sidecar centre={sidecar_centre.tolist()} scale={sidecar_scale}\n"
            + "\n".join(
                f"  {p}: delta_centre={d['delta_centre']:.6e} "
                f"delta_scale={d['delta_scale']:.6e}"
                for p, d in deltas.items()
            )
        )

    # Prefer the closest match deterministically (smallest combined delta).
    matched = min(
        matches,
        key=lambda p: deltas[p]["delta_centre"] + deltas[p]["delta_scale"],
    )
    matched_model, matched_verts, matched_desc = candidates[matched]

    joints_world = regress_joints(matched_model, matched_verts)

    # pose_id reports the physical pose (identity/rest in every candidate that matched);
    # mesh_topology disambiguates which construction produced the rendered mesh.
    pose_id = "rest"  # every matching candidate is the identity pose
    out = {
        "coco17_names": COCO17,
        "pose_id": pose_id,
        "matched_candidate": matched,
        "mesh_source": matched_desc,
        "mesh_topology": (
            "makehuman_basemesh_keep_unattached_19158"
            if matched == "rest_basemesh" else "soma"
        ),
        "joints_world": [[float(x) for x in row] for row in joints_world],
        "normalisation": {
            "centre": [float(x) for x in sidecar_centre],
            "scale": sidecar_scale,
        },
        "verify": {
            "sidecar_path": pathlib.Path(sidecars[0]).name,
            "sidecar_centre": [float(x) for x in sidecar_centre],
            "sidecar_scale": sidecar_scale,
            "delta_centre": deltas[matched]["delta_centre"],
            "delta_scale": deltas[matched]["delta_scale"],
            "match_tol": MATCH_TOL,
            "matches": matches,
            "candidates": {
                p: {
                    "delta_centre": d["delta_centre"],
                    "delta_scale": d["delta_scale"],
                    "n_vertices": d["n_vertices"],
                }
                for p, d in deltas.items()
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nmatched candidate: {matched}  (pose_id={pose_id}, matches={matches})")
    print(f"mesh source: {matched_desc}")
    print(f"wrote {out_path}")
    print("GT COCO-17 joints (ANNY world metres):")
    for name, row in zip(COCO17, joints_world):
        print(f"  {name:15s}  {row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
