# datasource-anny-render-corpus

The ANNY render corpus pipeline: ETNF schema, identity sampling, the canonical rigged model, the audits that gate a render, and the renderers.

## What it is for

It makes labelled training frames by rendering ANNY bodies, so the labels are true by construction rather than annotated. Every stage builds its body from the one rigged model in `anny_rig.py`, and the audits each carry a negative control that proves they can fail. It also holds the licence-filtered COCO person manifests; the val2017 manifest is the blinded holdout and is never trained on, tuned against or generated from. It also holds generated-synthetic tooling (image restyle and edit, edit scoring, voice cloning and speech recognition) in pixi environments of their own, apart from the constructed renders. The figures the code must keep true are tagged in `CLAIMS.md`.

## Build and run

The environments are pinned in `pixi.toml`.

```sh
pixi run -e anny preflight
pixi run -e corpus schema
```

## Licence

Either Apache-2.0 or MIT, at your option; see `LICENSE-APACHE` and `LICENSE-MIT`. A contribution is dual licensed the same way unless its author states otherwise. The licence covers the code. The data the scripts ingest and the models they drive keep their own terms, and anything generated here records the checkpoint that produced it.
