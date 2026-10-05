# Measured claims

Every figure here is a tagged claim. `check_readme_claims.py` re-derives each one from the live code and exits non-zero when one drifts, so a figure that changes with the code is changed here and nowhere else.

## Schema

`anny_render_schema.py` defines <!--claim:schema_relations=28 tol=0.5-->28 ETNF relations and <!--claim:schema_foreign_keys=29 tol=0.5-->29 foreign keys.

## Forearm twist

ANNY's stock rig leaves the forearm skin nearly still when the wrist turns. The corpus model re-weights the forearm as a linear elbow-to-wrist ramp landing on the wrist bone, so the ramp is the twist distribution, with no twist bone and no runtime step. RMSE against the anatomical ramp:

| rig                    | RMSE                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| stock                  | <!--claim:twist_rmse_stock_L=52.8 tol=3.0-->52.8°                                                          |
| wrist ramp             | <!--claim:twist_rmse_90_L=3.6 tol=1.0-->3.6° (left) / <!--claim:twist_rmse_90_R=4.1 tol=1.0-->4.1° (right) |
| no twist, the baseline | <!--claim:zero_twist_baseline=55.2 tol=0.5-->55.2°                                                         |

The re-weighting moves mass only between bones that are both at identity at rest, so the rest pose shifts by <!--claim:rest_pose_shift_mm=0.0 tol=0.001-->0.000 mm.

## Counts

ANNY has <!--claim:anny_bone_count=104 tol=0.5-->104 bones. The 100STYLE pose library has <!--claim:bvh_clip_count=810 tol=0.5-->810 clips. `interface_audit.py` names <!--claim:interfaces_total=17 tol=0.5-->17 interfaces, of which <!--claim:interfaces_unchecked=5 tol=0.5-->5 are reported as unchecked.
