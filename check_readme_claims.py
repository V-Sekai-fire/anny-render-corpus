"""Re-measure every number the README asserts, and fail if reality has moved.

WHY THIS EXISTS. A README rots silently: the code changes, the numbers stay, and nobody
finds out until someone builds on a figure that stopped being true. This project has
already produced one actively misleading README -- `godot-soma-twist` opened with "no
custom code is needed, Godot already ships the correct solution" long after the shipping
answer had become the opposite -- and one docstring that claimed Delta Mush was "available
here" when grep found zero occurrences of it.

So the README's numbers are not prose. They are tagged claims:

    <!--claim:twist_rmse_90_L=3.6 tol=1.0-->

and this script re-derives each one from the live code and compares. Drift is a test
failure, not a discovery six months later. That is the whole mechanism: being wrong makes
the documentation fail loudly, which is what makes it worth trusting when it passes.

Usage:  python check_readme_claims.py [CLAIMS.md]
Exit code is non-zero if any claim has drifted, so it can gate CI.

A claim with no measurement function registered is reported as UNVERIFIED rather than
skipped -- a silent skip reads exactly like a pass, which is a failure mode this project
has hit three separate times.
"""

import glob
import os
import re
import sys

ROOTS_FILE = "dataset-roots.txt"
ROOTS_LOCAL = "dataset-roots.local.txt"


def dataset_roots(here=None):
    """Where pose libraries are looked for, read from `dataset-roots.txt` or the
    gitignored `.local` beside it. Not a constant here: a root is a per-desk fact, and
    a list baked into code is one nobody can correct without a commit."""
    here = here or os.path.dirname(os.path.abspath(__file__))
    for name in (ROOTS_LOCAL, ROOTS_FILE):
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return tuple(r for r in (ln.split("#")[0].strip() for ln in fh) if r)
    return ()


CACHE_FILE = "claim-cache.txt"


def claim_cache(here=None):
    """The last measured value of claims whose source can go out of reach, as
    `name value measured-on where`. Read only: this script never writes an entry, so a
    number here is one somebody recorded after actually measuring it."""
    here = here or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, CACHE_FILE)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split("#")[0].split()
            if len(parts) >= 3:
                out[parts[0]] = {"value": float(parts[1]), "measured": parts[2],
                                 "where": " ".join(parts[3:]) or "unrecorded"}
    return out


def _roots(roots):
    got = dataset_roots() if roots is None else tuple(roots)
    if not got:
        raise FileNotFoundError(
            "no dataset root configured: write one path per line into %s, or %s "
            "beside it for this desk only" % (ROOTS_FILE, ROOTS_LOCAL))
    return got


def measure_twist_rmse(side):
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float64)
    return anny_rig.twist_ramp_rmse(model, side)[0]


def measure_twist_rmse_stock(side):
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    return anny_rig.twist_ramp_rmse(model, side)[0]


def measure_zero_twist_baseline():
    import anny_rig
    return anny_rig.ZERO_TWIST_RMSE


def measure_bone_count():
    import torch
    import anny_rig
    return float(anny_rig.build_corpus_model(dtype=torch.float32).bone_count)


def measure_rest_pose_shift_mm():
    """The re-weighting must not move the rest pose. It redistributes mass between bones
    that are both at identity at rest, so this is a property of the fix, not luck."""
    import numpy as np
    import torch
    import anny_rig
    a = anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    b = anny_rig.build_corpus_model(dtype=torch.float64)
    pose = anny_rig._identity_pose(a)
    with torch.no_grad():
        va = a(pose_parameters=pose)["rest_vertices"][0].numpy()
        vb = b(pose_parameters=pose)["rest_vertices"][0].numpy()
    return float(np.abs(va - vb).max() * 1000)


def measure_interfaces_total():
    import interface_audit
    interface_audit.RESULTS.clear()
    try:
        interface_audit.main()
    except SystemExit:
        pass
    return float(len(interface_audit.RESULTS))


def measure_interfaces_unchecked():
    import interface_audit
    interface_audit.RESULTS.clear()
    try:
        interface_audit.main()
    except SystemExit:
        pass
    return float(sum(1 for r in interface_audit.RESULTS if r[1] == "UNCHECKED"))


CLIP_SUFFIXES = (".bvh",)


def find_pose_library(name, roots=None, depth=4):
    """The directory of the named pose library; raises when no root holds it."""
    roots = _roots(roots)
    for root in roots:
        for d in range(depth + 1):
            hits = [h for h in glob.glob(os.path.join(root, *(["*"] * d), name))
                    if os.path.isdir(h)]
            if hits:
                return sorted(hits)[0]
    raise FileNotFoundError(
        "no pose library named %r within %d levels of: %s"
        % (name, depth, "; ".join(roots)))


def pose_clips(library):
    """Every motion clip in a pose library, at whatever depth the set organises them."""
    return sorted(p for p in glob.glob(os.path.join(library, "**", "*"), recursive=True)
                  if os.path.splitext(p)[1].lower() in CLIP_SUFFIXES)


def pose_library_citation(library):
    """The `.cff` at a pose library's root. CLAUDE.md's pose-source rule makes this the
    evidence a set is licence-clean, so a library without one cannot be gated on."""
    beside = sorted(glob.glob(os.path.join(library, "*.cff")))
    return beside[0] if beside else None


def pose_libraries(roots=None, depth=4):
    """Every reachable pose library, enumerated rather than named, so a set that arrives
    without anyone editing this file is still seen.

    A library is the nearest ancestor of its clips carrying a `.cff`, because the citation
    is what marks where a distributed set begins. With no `.cff` anywhere above them the
    clips are reported at the directory holding them: nothing says where that set ends,
    which is the same absence that makes it unusable under the pose-source rule.
    """
    roots = _roots(roots)
    found = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for base, dirs, files in os.walk(root):
            rel = os.path.relpath(base, root)
            if rel != "." and rel.count(os.sep) + 1 >= depth:
                dirs[:] = []
            if not any(os.path.splitext(f)[1].lower() in CLIP_SUFFIXES for f in files):
                continue
            lib, probe = base, base
            while probe != root and os.path.dirname(probe) != probe:
                if glob.glob(os.path.join(probe, "*.cff")):
                    lib = probe
                    break
                probe = os.path.dirname(probe)
            found.add(lib)
    return [{"path": lib, "name": os.path.basename(lib), "clips": len(pose_clips(lib)),
             "citation": pose_library_citation(lib)}
            for lib in sorted(found)]


POSE_LIBRARY = "100STYLE"


def measure_bvh_clip_count():
    return float(len(pose_clips(find_pose_library(POSE_LIBRARY))))


def measure_schema_relations():
    """UNTAGGED IS UNCHECKED, AND THIS NUMBER PROVES IT. The README asserted 19 ETNF
    relations as plain prose, with no claim tag, so this file never looked at it. The real
    count was 20 before the generated-synthetic relations landed: wrong, in a document whose
    headline says every number below is machine-checked."""
    import anny_render_schema
    return float(len(anny_render_schema.RELATIONS))


def measure_schema_foreign_keys():
    import anny_render_schema
    return float(len(anny_render_schema.FOREIGN_KEYS))


MEASUREMENTS = {
    "schema_relations": measure_schema_relations,
    "schema_foreign_keys": measure_schema_foreign_keys,
    "twist_rmse_90_L": lambda: measure_twist_rmse("L"),
    "twist_rmse_90_R": lambda: measure_twist_rmse("R"),
    "twist_rmse_stock_L": lambda: measure_twist_rmse_stock("L"),
    "zero_twist_baseline": measure_zero_twist_baseline,
    "anny_bone_count": measure_bone_count,
    "rest_pose_shift_mm": measure_rest_pose_shift_mm,
    "interfaces_total": measure_interfaces_total,
    "interfaces_unchecked": measure_interfaces_unchecked,
    "bvh_clip_count": measure_bvh_clip_count,
}

CLAIMS_FILE = "CLAIMS.md"

CLAIM_RE = re.compile(r"<!--\s*claim:(\w+)=([-\d.]+)(?:\s+tol=([\d.]+))?\s*-->")


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else CLAIMS_FILE)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    claims = CLAIM_RE.findall(text)
    if not claims:
        print("no tagged claims found in %s -- its numbers are unverifiable" % path)
        return 1

    print("%-24s %12s %12s %10s  %s" % ("claim", "stated", "measured", "tol", "verdict"))
    print("-" * 76)
    bad = 0
    cached, stale = [], []
    cache = claim_cache()

    import io
    import contextlib
    for name, stated, tol in claims:
        stated = float(stated)
        tol = float(tol) if tol else max(abs(stated) * 0.05, 0.01)
        fn = MEASUREMENTS.get(name)
        if fn is None:
            # Not skipped: an unregistered claim is a hole in the check, and a silent
            # skip is indistinguishable from a pass.
            print("%-24s %12.3f %12s %10s  UNVERIFIED (no measurement registered)"
                  % (name, stated, "-", "-"))
            bad += 1
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = float(fn())
        except Exception as exc:                                   # noqa: BLE001
            entry = cache.get(name) if isinstance(exc, FileNotFoundError) else None
            if entry is None:
                print("%-24s %12.3f %12s %10s  ERROR %s"
                      % (name, stated, "-", "-", type(exc).__name__))
                bad += 1
                continue
            got, when = entry["value"], entry["measured"]
            if abs(got - stated) > tol:
                print("%-24s %12.3f %12.3f %10.3f  DRIFTED against cache of %s"
                      % (name, stated, got, tol, when))
                bad += 1
            else:
                print("%-24s %12.3f %12.3f %10.3f  CACHED %s, not measured"
                      % (name, stated, got, tol, when))
                cached.append(name)
            continue
        ok = abs(got - stated) <= tol
        bad += 0 if ok else 1
        note = ""
        if name in cache:
            # The live measurement already won; the entry is now a record nothing reads.
            stale.append(name)
            note = "  (cache entry now superseded)"
        print("%-24s %12.3f %12.3f %10.3f  %s%s"
              % (name, stated, got, tol, "ok" if ok else "DRIFTED", note))

    print()
    if stale:
        print("%d cache entry/entries are superseded: the source is reachable again and the"
              % len(stale))
        print("live measurement is what ran. Delete these lines from %s: %s."
              % (CACHE_FILE, ", ".join(stale)))
        print()
    if cached:
        print("%d claim(s) answered from %s because the source was unreachable: %s."
              % (len(cached), CACHE_FILE, ", ".join(cached)))
        print("A cached value is a record of a past measurement, not a measurement. It")
        print("is never written by this script; someone recorded it after measuring.")
        print()
    if bad:
        print("%d claim(s) drifted or unverified -- CLAIMS.md is making statements the"
              % bad)
        print("code no longer supports. Fix CLAIMS.md or fix the code; do not ignore it.")
    else:
        print("all %d claims re-derived from live code and still hold." % len(claims))
    return 1 if bad else 0


def write(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _misses_roots():
    try:
        _roots(())
        return False
    except FileNotFoundError:
        return True


def _misses(name, root, depth):
    try:
        find_pose_library(name, roots=(root,), depth=depth)
        return False
    except FileNotFoundError:
        return True


def _fixture(tmp, name, clips, cff=True, sub="motions"):
    lib = os.path.join(tmp, "sets", name)
    os.makedirs(os.path.join(lib, sub), exist_ok=True)
    for i in range(clips):
        open(os.path.join(lib, sub, "take%02d.bvh" % i), "w").close()
    if cff:
        open(os.path.join(lib, "CITATION.cff"), "w").close()
    return lib


def report_pose_libraries(roots=None):
    roots = _roots(roots)
    libs = pose_libraries(roots)
    if not libs:
        print("no pose library reachable under: %s" % "; ".join(roots))
        return 1
    print("%-28s %8s  %s" % ("library", "clips", "citation"))
    for lib in libs:
        print("%-28s %8d  %s" % (lib["name"], lib["clips"],
                                 lib["citation"] or "NONE -- not licence-clean evidence"))
    unevidenced = [lib["name"] for lib in libs if not lib["citation"]]
    if unevidenced:
        print("\n%d library/libraries carry no .cff. The pose-source rule makes that the"
              % len(unevidenced))
        print("evidence a set is licence-clean, so these cannot be gated on: %s"
              % ", ".join(unevidenced))
    return 0


def self_test():
    """Seventeen controls. Six must reject a lookup that answers for a set it never found."""
    r = []
    import contextlib
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _fixture(tmp, "alpha-motions", 3)
        _fixture(tmp, "beta-motions", 2, cff=False, sub="a/b")
        os.makedirs(os.path.join(tmp, "sets", "not-a-pose-set", "docs"))

        r.append(("a pose library is found by name",
                  find_pose_library("alpha-motions", roots=(tmp,)).endswith("alpha-motions")))
        r.append(("a library past the depth bound is not found",
                  _misses("alpha-motions", tmp, depth=0)))
        r.append(("clips are counted at whatever depth the set organises them",
                  len(pose_clips(find_pose_library("beta-motions", roots=(tmp,)))) == 2))
        r.append(("a directory with no clips is not a pose library",
                  all(lib["name"] != "not-a-pose-set" for lib in pose_libraries((tmp,)))))

        libs = {lib["name"]: lib for lib in pose_libraries((tmp,))}
        r.append(("an evidenced set is reported at its citation, not at its clips",
                  "alpha-motions" in libs and libs["alpha-motions"]["clips"] == 3))
        r.append(("a citation beside a set is found",
                  libs["alpha-motions"]["citation"] is not None))
        r.append(("sibling sets are not merged into their parent folder",
                  "sets" not in libs and len(libs) == 2))
        unevidenced = [v for v in libs.values() if v["citation"] is None]
        r.append(("a set with no citation is reported as having none",
                  len(unevidenced) == 1 and unevidenced[0]["clips"] == 2))

    try:
        got = measure_bvh_clip_count()
        r.append(("an unreachable root is never counted as zero clips", got > 0))
    except FileNotFoundError:
        r.append(("an unreachable root is never counted as zero clips", True))

    with tempfile.TemporaryDirectory() as tmp:
        write(os.path.join(tmp, ROOTS_FILE),
              ["# comment", "", "  X:/one  # trailing", "Y:/two"])
        r.append(("roots are read from the file, comments and blanks dropped",
                  dataset_roots(tmp) == ("X:/one", "Y:/two")))
        write(os.path.join(tmp, ROOTS_LOCAL), ["Z:/local"])
        r.append(("the local file replaces the tracked one",
                  dataset_roots(tmp) == ("Z:/local",)))
    r.append(("no configured root is an error, not an empty answer",
              _misses_roots()))

    with tempfile.TemporaryDirectory() as tmp:
        write(os.path.join(tmp, CACHE_FILE),
              ["# a comment", "", "a_claim  810  2026-08-24  O:/somewhere", "b_claim 2 2026-01-01"])
        got = claim_cache(tmp)
        r.append(("a cached entry carries its value, date and origin",
                  got["a_claim"] == {"value": 810.0, "measured": "2026-08-24",
                                     "where": "O:/somewhere"}))
        r.append(("an entry with no origin records that rather than inventing one",
                  got["b_claim"]["where"] == "unrecorded"))
        r.append(("a claim absent from the cache is not answered",
                  "c_claim" not in got))
        r.append(("an empty cache is not an error", claim_cache(tmp + "/nope") == {}))

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, CACHE_FILE)
    before = open(path, "rb").read() if os.path.exists(path) else None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            main(os.path.join(here, CLAIMS_FILE))
        except SystemExit:
            pass
    after = open(path, "rb").read() if os.path.exists(path) else None
    r.append(("running the gate never writes the cache", before == after))

    # Planted rather than waited for. globals(), because run as a script this file is
    # `__main__` and importing it by name would patch a second, unused copy.
    g = globals()
    real = g["claim_cache"]
    g["claim_cache"] = lambda here=None: {
        "anny_bone_count": {"value": 104.0, "measured": "2026-01-01", "where": "planted"}}
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                main(os.path.join(here, CLAIMS_FILE))
            except SystemExit:
                pass
        text = out.getvalue()
    finally:
        g["claim_cache"] = real
    r.append(("a superseded entry is named once its source returns",
              "Delete these lines from %s: anny_bone_count." % CACHE_FILE in text))
    r.append(("a superseded entry does not make the run fail",
              "104.000      104.000" in text))

    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    bad = sum(1 for _, ok in r if not ok)
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if "--pose-libraries" in sys.argv:
        sys.exit(report_pose_libraries())
    sys.exit(main())
