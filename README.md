# anny-render-corpus

The ANNY render corpus pipeline: schema, identity sampling, the canonical rigged model,
the audits that gate a render run, and the 100STYLE pose reader.

**Every number below is machine-checked.** Run `python check_readme_claims.py` — it
re-derives each figure from the live code and exits non-zero on drift. If this README is
wrong, that command says so. That is the point: a document that fails loudly when it stops
being true is worth trusting when it passes.

```
python check_readme_claims.py     # verify this README against reality
python preflight_audit.py <corpus>  # gate before a render run (full corpus, ~95 s)
python interface_audit.py         # the pipeline's edges
python test_preflight.py <corpus> # red/green: prove every check can fail
```

## What is here

| file | what it does |
|---|---|
| `anny_render_schema.py` | <!--claim:schema_relations=28 tol=0.5-->28 ETNF relations, <!--claim:schema_foreign_keys=29 tol=0.5-->29 foreign keys, `validate()`, deterministic ids |
| `anny_rig.py` | **the canonical model.** Every stage builds from here, never bare `anny.Anny` |
| `sample_identities.py` | 23,000 identities (22,511 train / 489 val) |
| `preflight_audit.py` | 29 semantic/physical checks; gates the render run |
| `test_preflight.py` | red/green — every check must be *able* to fail |
| `interface_audit.py` | the 17 named interfaces between components |
| `corpus_defect_rate.py` | quality as an exceedance **rate**, not a mean |
| `bvh_parse.py` | 100STYLE BVH reader + FK, no retarget opinions |
| `bvh_retarget_probe.py` | which retarget formulation transfers a pose — none clears the bind-orientation floor |
| `coco_zip_to_etnf.py`, `filter_coco_licenses.py` | COCO ingest, license filtering |

## The rig fix, in one line

ANNY's stock rig cannot transmit forearm twist: driving the wrist — the only channel motion
capture supplies — leaves the forearm skin nearly still. The corpus re-weights the forearm
as a linear elbow→wrist ramp landing on the **wrist bone**, so the ramp itself becomes the
twist distribution. **No twist bone, no runtime step.**

| | RMSE vs the anatomical ramp |
|---|---|
| stock rig | <!--claim:twist_rmse_stock_L=52.8 tol=3.0-->52.8° |
| **shipping (wrist ramp)** | <!--claim:twist_rmse_90_L=3.6 tol=1.0-->3.6° (L) / <!--claim:twist_rmse_90_R=4.1 tol=1.0-->4.1° (R) |
| "no twist at all" baseline | <!--claim:zero_twist_baseline=55.2 tol=0.5-->55.2° |

The re-weighting is provably rest-neutral — it only moves mass between bones that are both
at identity at rest: <!--claim:rest_pose_shift_mm=0.0 tol=0.001-->0.000 mm.

ANNY has <!--claim:anny_bone_count=104 tol=0.5-->104 bones.
100STYLE has <!--claim:bvh_clip_count=810 tol=0.5-->810 clips across 100 styles.

## Interfaces, not components

`interface_audit.py` names <!--claim:interfaces_total=17 tol=0.5-->17 interfaces, of which
<!--claim:interfaces_unchecked=5 tol=0.5-->5 are still UNCHECKED and reported loudly.

Every defect this project has hit lived at a boundary, never inside a component. Full list
and the recurring failure modes: **`weftspun/logbook/PITFALLS.md`**.

## Superseded — claims that do not hold

This section grows. It carries the most weight in the document: a reader who knows the dead
ends is better off than one who knows only the current answer.

| claim | why it does not hold |
|---|---|
| "no ratio works; RMS ~39° at every ratio" | measures about **world Z** rather than the bone roll axis — the local→world map is the identity, so "local Z" sits 55° off the forearm, and the run measures a bend rather than a pronation. |
| "rig=soma returns mesh and skeleton in different frames" | compares a **rest** skeleton against an **identity-pose** mesh. Paired correctly, containment reaches 100% for every rig and phenotype. |
| "~9.7° size-correlated thigh error" | a weak observable. Centroid direction reads +9.7° where principal-axis reads −4.0° on the same runs. Joint angle gives thigh 2.2°, arms <1°. |
| "neither BVH formulation transfers the pose" | rests on a 153 mm residual with **no baseline**. Two *rest* skeletons score 139.7 mm. `local` sits on the floor; the blocker is bind orientation. |
| "Delta Mush is available in `anny_rig`" | no such code exists there. `grep` finds zero occurrences. |

## Falsifiers

What would show these answers do not hold:

- **Twist fix** — a pronation angle where the wrist ramp exceeds ~15° RMSE, or an L/R
  asymmetry above 3°. Both are gated in `preflight_audit.py`.
- **Audit power** — a defect affecting fewer identities than the stated detection floor.
  The audit prints its own floor (43 ppm on a full decode) and **fails** if asked to
  certify below what its sample size can resolve.
- **Skeleton/mesh pairing** — any joint sitting >1% of stature from the nearest vertex at
  an extreme phenotype. Checked with a negative control that must reject the mispairing.

## COCO lineage

COCO person images (license-filtered, commercial-safe) → GEM-X →
SOMA-X pose + identity coefficients. feed the earlier line of work, which remains here (`filter_coco_licenses.py`:
val2017 523/5,000; train2017 12,620/118,287) ) and is right-sized for **evaluation and domain adaptation** rather than from-scratch
training. The training-scale source is self-generated synthetic ANNY renders, which the rest
of this repo builds.

## VRM avatars and mask checks

A pilot render of VRM avatars (COCO instance masks) and two ways to judge a person mask.

| tool | how to run it |
|---|---|
| `vrm_fetch.py`, `vrm_load.py`, `vrm_render.py` | `pixi run vrm-pilot`: fetches the CC0 avatars in `vrm_models.tsv`, checks that the aux pass leaves RGB and masks unchanged (`vrm-check-aux`), renders 300 images with `--aux`, then verifies the COCO file and writes a contact sheet |
| `depth_agreement.py` | MoGe-3 depth agreement, the mask acceptance check. `infer` needs the Windows env `C:\Users\ernest.lee\AppData\Local\moge3\pixi.toml`; `score` / `compare` / `sheet` run in that env too: `pixi run --manifest-path C:/Users/ernest.lee/AppData/Local/moge3/pixi.toml python depth_agreement.py <subcommand>` |
| `matte_refine.py` | tightens RF-DETR person masks with a BiRefNet_HR alpha matte, scored by `depth_agreement`: `pixi run --manifest-path C:/Users/ernest.lee/AppData/Local/matting-hr/pixi.toml python matte_refine.py` |
| `score_masks.py` | EditScore as a mask grader (parked, see below): `pixi run -e editscore python score_masks.py --rfdetr ... --frames ... --frame-dir ... --out data/score_masks/vrchat23 --sheet sheet.png` |

## Open work

Tracked as issues, not prose — see this repo's issue list. Critical path is **#1**
(100STYLE bind-orientation correction) → poses → scenes → rung 0 of the render ladder.

## Parked

Recorded so it is not lost, and not scheduled.

- Shelved 2026-09-27: the ~5k-image VRM render corpus with a train/val split by avatar.
  A pilot of 300 images from 25 CC0 avatars exists (`pixi run vrm-pilot`). Scaling it is
  parked because no vehicle needs avatar-person detection; faces are RFD 2262's Car.
  Unpark when a vehicle needs it.
- Shelved 2026-09-27: EditScore as a mask judge (`score_masks.py`). On 23 VRChat frames
  it does not track mask quality: its best pick agrees with the MoGe-3 depth check on 6 of
  23, and `--fast` is 12x faster but reaches only Spearman 0.42–0.52 against the full
  mode. Acceptance uses `depth_agreement.py` instead. Calibration on the pilot renders
  (mode a, AUC against IoU) has not been run. Unpark as part of RFD 2262's "grade that
  teaches" (a compact retrained EditScore), not before.
- Shelved 2026-09-27: the pilot renders' known gaps. Pose randomisation is light (many
  avatars near T-pose); alpha-blended materials are refused; renders are not bit-exact,
  because single-thread drjit deadlocks with the MToon integrator; Mitsuba CUDA is
  unavailable under WSL (OptiX). Unpark with the VRM render corpus above.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

This covers the **code** in this repository. It does not relicense the data the scripts
ingest or the models they drive, each of which keeps its own terms: COCO images are filtered
to commercial-and-derivatives-safe licences by `filter_coco_licenses.py`, and OmniGen2,
EditScore and RF-DETR are Apache-2.0 upstream. Anything generated here records the
checkpoint that produced it, per CLAUDE.md's condition 1.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you shall be dual licensed as above, without any additional terms or
conditions.
