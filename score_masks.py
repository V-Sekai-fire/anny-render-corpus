"""Grade person/avatar mask proposals with EditScore, and calibrate the grade against IoU.

Step 2 of the RF-DETR-avatar plan. A mask is shown to EditScore as an *edit*: the source is
the frame, the edited image is the same frame with the candidate mask filled translucent
magenta plus a 2 px outline, and the instruction asks to highlight exactly the whole character.
A mask that covers the whole avatar and nothing else is the edit that follows the instruction;
a bitten, bloated, shifted or wrong-object mask is not.

TWO MODES.

  (a) --coco ann.json --images DIR [--corrupt]
      Ground-truth renders. Every GT instance is scored, and with --corrupt five deliberately
      wrong variants of it too (eroded 15 %, dilated 15 %, shifted 10 % of the bbox, the wrong
      instance or a background blob, the top half). Each row carries IoU against GT, and the
      summary gives the AUC of `overall` for separating IoU >= 0.8 from IoU < 0.8, and the
      lowest tau whose accepted set is >= 90 % precise. The plan's gate: AUC < 0.8 means drop
      the pseudo-labels and train on renders only.

  (b) --rfdetr results.bin --frames frames.txt --frame-dir DIR
      Real captures. results.bin is N records of float32 boxes(100,4) cx,cy,w,h normalised,
      logits(100,91), mask logits(100,78,78) over the full frame, in frames.txt order; person
      is class 1. Candidates are the top-k queries by sigmoid(person logit) x box area; the
      mask is the bilinear upsample of the logits, thresholded at sigmoid > 0.5.

Every candidate is scored under two phrasings of the instruction and both are kept, because a
grader whose verdict flips with the wording is not measuring the mask.

The loader is score_edits.py's: Qwen3-VL-8B + the EditScore LoRA at NF4, adapter attached rather
than merged, images capped at 512x512-equivalent (the configuration measured to fit 8 GB).
The overlay is drawn AFTER the cap so the 2 px outline survives the downscale.

    pixi run -e editscore python score_masks.py --rfdetr ... --frames ... --frame-dir ... \\
        --out data/score_masks/vrchat23 --sheet sheet.png
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

INSTRUCTIONS = {
    "primary": "Highlight exactly the whole character (the person or avatar), nothing else.",
    "alt": "Color over the entire body of the character (the person or avatar) and nothing "
           "else in the scene.",
}
FILL = (255, 0, 255)
ALPHA = 0.45
IOU_GOOD = 0.8

REC_BOXES, REC_LOGITS, MASK_S = 100 * 4, 100 * 91, 78


# ---------------------------------------------------------------------------- images

def cap(im, max_pixels):
    if not max_pixels or im.width * im.height <= max_pixels:
        return im
    s = (max_pixels / (im.width * im.height)) ** 0.5
    return im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)


def resize_mask(mask, size):
    """Nearest-neighbour resize of a bool mask to (W, H)."""
    if mask.shape[::-1] == tuple(size):
        return mask
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def overlay(im, mask):
    """Frame with the mask filled translucent magenta plus a 2 px outline."""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    a[m] = a[m] * (1 - ALPHA) + np.array(FILL, np.float32) * ALPHA
    mi = Image.fromarray(m.astype(np.uint8) * 255)
    # 2 px ring just inside the mask boundary: mask minus its 5x5 erosion.
    ring = (np.asarray(mi) > 127) & ~(np.asarray(mi.filter(ImageFilter.MinFilter(5))) > 127)
    a[ring] = FILL
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


# ---------------------------------------------------------------------------- COCO masks
# pycocotools is not in the editscore env, so decode the three segmentation forms here.

def rle_string_to_counts(s):
    """COCO's compressed RLE string (LEB128-like, 5 bits per char, delta from counts[i-2])."""
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_segmentation(seg, h, w):
    if isinstance(seg, list):                              # polygons
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for poly in seg:
            if len(poly) >= 6:
                d.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        return np.asarray(im) > 0
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = rle_string_to_counts(counts)
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(w, h).T                            # COCO RLE is column-major


# ---------------------------------------------------------------------------- corruptions

def morph_to_area(mask, target_ratio, grow, max_iter=200):
    """Erode (grow=False) or dilate (grow=True) by 1 px steps until area crosses target."""
    area0 = mask.sum()
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    f = ImageFilter.MaxFilter(3) if grow else ImageFilter.MinFilter(3)
    cur = mask
    for _ in range(max_iter):
        im = im.filter(f)
        cur = np.asarray(im) > 127
        r = cur.sum() / max(area0, 1)
        if (grow and r >= target_ratio) or (not grow and r <= target_ratio) or cur.sum() == 0:
            break
    return cur


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def shifted(mask, frac=0.10):
    x0, y0, x1, y1 = bbox_of(mask)
    dx, dy = int(round(frac * (x1 - x0))), int(round(frac * (y1 - y0)))
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[dy:, dx:] = mask[:h - dy, :w - dx]
    return out


def half(mask):
    x0, y0, x1, y1 = bbox_of(mask)
    out = mask.copy()
    out[(y0 + y1) // 2:, :] = False
    return out


def background_blob(mask):
    """An ellipse the size of the GT bbox, mirrored across the frame, minus the GT."""
    h, w = mask.shape
    x0, y0, x1, y1 = bbox_of(mask)
    bw, bh = x1 - x0, y1 - y0
    cx = w - (x0 + x1) / 2
    if abs(cx - (x0 + x1) / 2) < bw:                       # centred subject: go to a side
        cx = bw / 2 if (x0 + x1) / 2 > w / 2 else w - bw / 2
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).ellipse((cx - bw / 2, y0, cx + bw / 2, y1), fill=1)
    return (np.asarray(im) > 0) & ~mask


def corruptions(gt, others):
    out = [("eroded15", morph_to_area(gt, 0.85, grow=False)),
           ("dilated15", morph_to_area(gt, 1.15, grow=True)),
           ("shifted10", shifted(gt)),
           ("half", half(gt))]
    wrong = [o for o in others if iou(o, gt) < 0.5 and o.sum() > 0]
    if wrong:
        out.append(("wrong_instance", max(wrong, key=lambda m: m.sum())))
    else:
        out.append(("background_blob", background_blob(gt)))
    return out


# ---------------------------------------------------------------------------- scorer

def load_scorer(precision, num_pass):
    if precision == "nf4":
        from score_edits import patch_for_4bit
        patch_for_4bit()
    import torch
    from editscore import EditScore
    from score_edits import BASE, ADAPTER
    t0 = time.time()
    scorer = EditScore(backbone="qwen3vl", model_name_or_path=BASE, lora_path=ADAPTER,
                       score_range=25, num_pass=num_pass)
    print(f"loaded in {time.time() - t0:.0f}s | {precision} | weights "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return scorer


def grade(scorer, src, edited):
    import torch
    out = {}
    for key, text in INSTRUCTIONS.items():
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            r = scorer.evaluate([src, edited], text)
            out[key] = {k: float(r[k]) for k in
                        ("overall", "prompt_following", "consistency", "perceptual_quality")}
            out[key]["reasoning"] = {"SC": r.get("SC_reasoning"), "PQ": r.get("PQ_reasoning")}
        except Exception as exc:                           # a non-answer, not a low score
            out[key] = {"overall": None, "error": f"{type(exc).__name__}: {exc}"}
        out[key]["seconds"] = round(time.time() - t, 2)
        out[key]["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


# ---------------------------------------------------------------------------- profiling

def _find(net, cls_name=None, suffix=None):
    hits = [(n, m) for n, m in net.named_modules()
            if (cls_name and type(m).__name__ == cls_name) or (suffix and n.endswith(suffix))]
    return min(hits, key=lambda h: len(h[0]))[1]


def profile_evaluate(scorer, src, edited, instruction):
    """One evaluate() split into its stages with CUDA-synchronised wall timers.

    Mirrors EditScore.evaluate for num_pass=1: SC prompt (source + edited, instruction) and PQ
    prompt (edited only), each prepared by the processor then sampled by generate(). Hooks on
    the vision tower and on the LM forward time every call; the first LM forward of a generate
    is the prefill (vision tower included, reported separately), every later one a decode step.
    """
    import torch
    m, net = scorer.model, scorer.model.model
    visual = _find(net, suffix="visual")
    lm = _find(net, cls_name="Qwen3VLForConditionalGeneration")
    sync = torch.cuda.synchronize
    log = []

    def pre(tag):
        def f(*_a, **_k):
            sync()
            log.append((tag, "start", time.perf_counter()))
        return f

    def post(tag):
        def f(*_a, **_k):
            sync()
            log.append((tag, "end", time.perf_counter()))
        return f

    hooks = [visual.register_forward_pre_hook(pre("vision")),
             visual.register_forward_hook(post("vision")),
             lm.register_forward_pre_hook(pre("lm")),
             lm.register_forward_hook(post("lm"))]
    image_token = m.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    prompts = {"SC": ([src, edited], scorer.SC_prompt.replace("<instruction>", instruction)),
               "PQ": (edited, scorer.PQ_prompt)}
    out = {}
    try:
        for name, (imgs, text) in prompts.items():
            torch.cuda.reset_peak_memory_stats()
            sync()
            t0 = time.perf_counter()
            inputs = m.prepare_input(imgs, text)
            sync()
            t1 = time.perf_counter()
            log.clear()
            from editscore.mllm_tools.qwen3vl import set_seed
            set_seed(scorer.seed)
            with torch.no_grad():
                ids = net.generate(**inputs, max_new_tokens=512, do_sample=True,
                                   temperature=m.temperature, top_p=0.9, top_k=20)
            sync()
            t2 = time.perf_counter()
            spans = {"vision": [], "lm": []}
            opened = {}
            for tag, kind, t in log:
                if kind == "start":
                    opened[tag] = t
                else:
                    spans[tag].append(t - opened.pop(tag))
            vision = sum(spans["vision"])
            n_prompt = int(inputs["input_ids"].shape[1])
            gen = ids[0, n_prompt:]
            text_out = m.processor.decode(gen, skip_special_tokens=True)
            decode_fwd = sum(spans["lm"][1:])
            out[name] = {
                "images": len(imgs) if isinstance(imgs, list) else 1,
                "prompt_tokens": n_prompt,
                "image_tokens": int((inputs["input_ids"] == image_token).sum()),
                "generated_tokens": int(gen.numel()),
                "preprocess_s": round(t1 - t0, 4),
                "vision_encoder_s": round(vision, 4),
                "prefill_lm_s": round(spans["lm"][0] - vision, 4),
                "decode_s": round((t2 - t1) - spans["lm"][0], 4),
                "decode_forward_s": round(decode_fwd, 4),
                "decode_ms_per_token": round(1000 * decode_fwd / max(len(spans["lm"]) - 1, 1), 2),
                "generate_total_s": round(t2 - t1, 4),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "output": text_out,
            }
    finally:
        for h in hooks:
            h.remove()
    total = sum(v["preprocess_s"] + v["generate_total_s"] for v in out.values())
    out["evaluate_total_s"] = round(total, 3)
    out["share"] = {
        stage: round(sum(v[key] for k, v in out.items() if k in prompts) / total, 3)
        for stage, key in (("preprocess", "preprocess_s"), ("vision", "vision_encoder_s"),
                           ("prefill", "prefill_lm_s"), ("decode", "decode_s"))}
    return out


# ---------------------------------------------------------------------------- fast mode
# The decider design (ollaya docs/families/decider.md): one prefill plus a readout, no
# generation. The profile says why -- ~92 % of an evaluate() is sampling ~100-token reasonings.
#
# 1. The assistant turn is forced up to the score slot, `{"reasoning": "", "score": [`, and the
#    next-token logits are read there. Qwen splits numbers into digits ("25" = "2","5"), so a
#    score in 0..25 is: P(first digit) at the slot, then one extra 1-token forward after "1" and
#    after "2" for P(second digit | terminator); the KV cache is cropped back after each branch.
#    The expected value over 0..25 is the score; the mode is appended to reach the second slot.
# 2. PQ does not mention the instruction, so it is read once per mask, not once per phrasing.
# 3. SC's prefix (chat header + the frame) is identical for every candidate and both phrasings:
#    it is prefilled once per frame, its cache repeated across the candidates, and the
#    candidates' suffixes (overlay + instruction + forced prefix) -- equal length, because the
#    overlays share the frame's size -- run as one batch.
# M-RoPE positions are computed over the whole sequence by get_rope_index and sliced, so the
# split prefill sees exactly the positions a single prefill would.

# The EditScore LoRA opens every answer with an empty think block; the forced prefix copies the
# format the full mode generates (see profile.json outputs), with the reasoning left empty.
FORCED = '<think>\n\n</think>\n\n{\n"reasoning" : "",\n"score" : ['


class FastScorer:
    def __init__(self, scorer, temperature=1.0):
        import torch
        self.torch = torch
        self.s = scorer
        self.proc = scorer.model.processor
        self.lm = _find(scorer.model.model, cls_name="Qwen3VLForConditionalGeneration")
        tok = self.proc.tokenizer
        self.digit = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        for d, i in enumerate(self.digit):
            assert tok.decode([i]) == str(d), f"digit {d} is not a single token"
        self.comma = tok.encode(",")[0]
        self.close = tok.encode("]")[0]
        self.vision_end = tok.convert_tokens_to_ids("<|vision_end|>")
        self.T = temperature
        self.range = scorer.score_range

    # -- tokenisation -------------------------------------------------------------------
    def encode(self, images, text):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": text}]}]
        s = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[s + FORCED], images=images, return_tensors="pt").to("cuda")
        ids, grid, mm = enc["input_ids"], enc["image_grid_thw"], enc["mm_token_type_ids"]
        vis, _ = self.lm.model.get_rope_index(ids, mm_token_type_ids=mm, image_grid_thw=grid)
        text_pos = self.torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1)
        enc["pos4"] = self.torch.cat([text_pos, vis.to(ids.device)], 0)
        return enc

    def forward(self, ids, pos4, cache, **vision):
        out = self.lm(input_ids=ids, position_ids=pos4, past_key_values=cache, use_cache=True,
                      logits_to_keep=1, **vision)
        return out.logits[:, -1].float(), out.past_key_values

    # -- cache helpers ------------------------------------------------------------------
    @staticmethod
    def _shallow(cache):
        import copy
        c = copy.copy(cache)
        c.layers = [copy.copy(l) for l in cache.layers]
        return c

    def repeat(self, cache, n):
        c = self._shallow(cache)
        c.batch_repeat_interleave(n)
        return c

    def row(self, cache, i):
        c = self._shallow(cache)
        for l in c.layers:
            l.keys, l.values = l.keys[i:i + 1], l.values[i:i + 1]
        return c

    # -- the readout --------------------------------------------------------------------
    def read_pair(self, logits, cache, t_next, v_next):
        """Two scores for one row, starting at the first score slot. Returns (E1, E2, raw)."""
        torch = self.torch
        state = {"t": t_next, "v": v_next, "cache": cache}

        def step(tokens, keep):
            n = len(tokens)
            ids = torch.tensor([tokens], device="cuda")
            t = torch.arange(state["t"], state["t"] + n, device="cuda")
            v = torch.arange(state["v"], state["v"] + n, device="cuda")
            pos = torch.stack([t, v, v, v]).view(4, 1, n)
            lg, state["cache"] = self.forward(ids, pos, state["cache"])
            if keep:
                state["t"] += n
                state["v"] += n
            else:
                state["cache"].crop(-n)
            return lg[0]

        raws, exps = [], []
        for term in (self.comma, self.close):
            first = logits[self.digit]
            after = {d: step([self.digit[d]], keep=False)[self.digit + [term]] for d in (1, 2)}
            raw = {"first": first.tolist(), "after1": after[1].tolist(),
                   "after2": after[2].tolist()}
            e, mode = self.expect(raw)
            raws.append(raw)
            exps.append(e)
            nxt = [self.digit[int(c)] for c in str(mode)] + ([self.comma] if term == self.comma
                                                            else [])
            if term == self.comma:
                nxt += self.proc.tokenizer.encode(" ")
                logits = step(nxt, keep=True)
        return exps[0], exps[1], raws

    def expect(self, raw, T=None):
        """Distribution over 0..score_range from the three restricted logit vectors."""
        T = self.T if T is None else T
        return expect_from_raw(raw, self.range, T)

    # -- one frame ----------------------------------------------------------------------
    def score_frame(self, src, overlays):
        torch = self.torch
        B = len(overlays)
        res = [dict() for _ in range(B)]
        with torch.no_grad():
            # PQ: one batch of B full sequences.
            encs = [self.encode([ov], self.s.PQ_prompt) for ov in overlays]
            e0 = encs[0]
            L = e0["input_ids"].shape[1]
            lg, cache = self.forward(
                e0["input_ids"].repeat(B, 1), e0["pos4"].repeat(1, B, 1), None,
                pixel_values=torch.cat([e["pixel_values"] for e in encs]),
                image_grid_thw=e0["image_grid_thw"].repeat(B, 1))
            v_next = int(e0["pos4"][1:, 0, -1].max()) + 1
            for i in range(B):
                a, b, raw = self.read_pair(lg[i], self.row(cache, i), L, v_next)
                res[i]["PQ"] = (a, b, raw)
            del cache
            # SC: shared prefix (header + frame), batched suffixes per phrasing.
            prefix_cache, cut = None, None
            for key, text in INSTRUCTIONS.items():
                sc_text = self.s.SC_prompt.replace("<instruction>", text)
                encs = [self.encode([src, ov], sc_text) for ov in overlays]
                e0 = encs[0]
                ids, pos4, grid = e0["input_ids"], e0["pos4"], e0["image_grid_thw"]
                if prefix_cache is None:
                    cut = int((ids[0] == self.vision_end).nonzero()[0]) + 1
                    n1 = int(grid[0].prod())
                    _, prefix_cache = self.forward(
                        ids[:, :cut], pos4[..., :cut], None,
                        pixel_values=e0["pixel_values"][:n1], image_grid_thw=grid[:1])
                n1 = int(grid[0].prod())
                lg, cache = self.forward(
                    ids[:, cut:].repeat(B, 1), pos4[..., cut:].repeat(1, B, 1),
                    self.repeat(prefix_cache, B),
                    pixel_values=torch.cat([e["pixel_values"][n1:] for e in encs]),
                    image_grid_thw=grid[1:2].repeat(B, 1))
                L = ids.shape[1]
                v_next = int(pos4[1:, 0, -1].max()) + 1
                for i in range(B):
                    a, b, raw = self.read_pair(lg[i], self.row(cache, i), L, v_next)
                    res[i][key] = (a, b, raw)
                del cache
        out = []
        k = self.range / 10
        for r in res:
            pq1, pq2, pq_raw = r["PQ"]
            scores = {}
            for key in INSTRUCTIONS:
                pf, cons, sc_raw = r[key]
                sc, pq = min(pf, cons) / k, min(pq1, pq2) / k
                scores[key] = {"overall": float((sc * pq) ** 0.5),
                               "prompt_following": pf / k, "consistency": cons / k,
                               "perceptual_quality": pq, "raw_logits": {"SC": sc_raw}}
            scores["PQ_raw_logits"] = pq_raw
            out.append(scores)
        return out


def expect_from_raw(raw, score_range=25, T=1.0):
    def sm(x):
        x = np.asarray(x, np.float64) / T
        e = np.exp(x - x.max())
        return e / e.sum()
    p1, a1, a2 = sm(raw["first"]), sm(raw["after1"]), sm(raw["after2"])
    p = np.zeros(score_range + 1)
    for d in range(10):
        if d == 1:
            p[1] += p1[1] * a1[-1]
            for j in range(10):
                if 10 + j <= score_range:
                    p[10 + j] += p1[1] * a1[j]
        elif d == 2:
            p[2] += p1[2] * a2[-1]
            for j in range(10):
                if 20 + j <= score_range:
                    p[20 + j] += p1[2] * a2[j]
        elif d <= score_range:
            p[d] += p1[d]
    p /= p.sum()
    return float((p * np.arange(score_range + 1)).sum()), int(p.argmax())


def compare_full(rows, full_path):
    """Fast vs the full generated scores: Spearman, max |diff|, and a fitted temperature."""
    full = {}
    for l in open(full_path):
        r = json.loads(l)
        full[(os.path.basename(r["image"]), r["kind"], r.get("instance"))] = r
    out = {}
    for key in INSTRUCTIONS:
        pairs = []
        for r in rows:
            f = full.get((os.path.basename(r["image"]), r["kind"], r.get("instance")))
            if f and f["scores"][key].get("overall") is not None:
                pairs.append((r, f["scores"][key]["overall"]))
        if len(pairs) < 3:
            continue
        ref = np.array([v for _, v in pairs])

        def overall_at(T):
            vals = []
            for r, _ in pairs:
                sc = r["scores"][key]["raw_logits"]["SC"]
                pq = r["scores"]["PQ_raw_logits"]
                e = [expect_from_raw(x, T=T)[0] / 2.5 for x in sc + pq]
                vals.append((min(e[0], e[1]) * min(e[2], e[3])) ** 0.5)
            return np.array(vals)

        fast1 = overall_at(1.0)
        grid = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        mse = {T: float(((overall_at(T) - ref) ** 2).mean()) for T in grid}
        Tbest = min(mse, key=mse.get)
        fb = overall_at(Tbest)
        out[key] = {"n": len(pairs), "spearman": spearman(fast1, ref),
                    "max_abs_diff": float(np.abs(fast1 - ref).max()),
                    "mean_abs_diff": float(np.abs(fast1 - ref).mean()),
                    "best_T": Tbest, "mse_by_T": mse,
                    "max_abs_diff_at_best_T": float(np.abs(fb - ref).max()),
                    "mean_abs_diff_at_best_T": float(np.abs(fb - ref).mean())}
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------- candidates

def candidates_coco(args):
    coco = json.load(open(args.coco))
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    n = 0
    for img in coco["images"]:
        if args.limit and n >= args.limit:
            break
        n += 1
        path = os.path.join(args.images, img["file_name"])
        full = Image.open(path).convert("RGB")
        src = cap(full, args.max_pixels)
        masks = [resize_mask(decode_segmentation(a["segmentation"], img["height"], img["width"]),
                             src.size)
                 for a in anns.get(img["id"], [])]
        for i, gt in enumerate(masks):
            if gt.sum() == 0:
                continue
            base = {"image": path, "image_id": img["id"], "instance": i}
            cands = [("gt", gt)]
            if args.corrupt:
                cands += corruptions(gt, masks[:i] + masks[i + 1:])
            for kind, m in cands:
                yield src, m, dict(base, kind=kind, iou=round(iou(m, gt), 4),
                                   area=int(m.sum()))


def candidates_rfdetr(args):
    names = [l.strip().replace("\\", "/").split("/")[-1] for l in open(args.frames) if l.strip()]
    rec = REC_BOXES + REC_LOGITS + 100 * MASK_S * MASK_S
    data = np.fromfile(args.rfdetr, dtype=np.float32)
    assert data.size == rec * len(names), (data.size, rec, len(names))
    sig = lambda x: 1 / (1 + np.exp(-x))
    for fi, name in enumerate(names[:args.limit or None]):
        r = data[fi * rec:(fi + 1) * rec]
        boxes = r[:REC_BOXES].reshape(100, 4)
        logits = r[REC_BOXES:REC_BOXES + REC_LOGITS].reshape(100, 91)
        mlog = r[REC_BOXES + REC_LOGITS:].reshape(100, MASK_S, MASK_S)
        ps = sig(logits[:, 1])
        rank = ps * boxes[:, 2] * boxes[:, 3]
        path = os.path.join(args.frame_dir, name.rsplit(".", 1)[0] + ".png")
        src = cap(Image.open(path).convert("RGB"), args.max_pixels)
        for k, q in enumerate(np.argsort(-rank)[:args.topk]):
            up = np.asarray(Image.fromarray(mlog[q].astype(np.float32), "F")
                            .resize(src.size, Image.BILINEAR))
            m = up > 0.0                                   # sigmoid > 0.5
            yield src, m, {"image": path, "frame": fi, "kind": f"rfdetr_top{k + 1}",
                           "query": int(q), "person_p": round(float(ps[q]), 4),
                           "argmax_class": int(logits[q].argmax()),
                           "box_cxcywh": [round(float(v), 4) for v in boxes[q]],
                           "area": int(m.sum()), "iou": None}


# ---------------------------------------------------------------------------- summary

def auc(pos, neg):
    """Mann-Whitney AUC with ties counted half."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def calibrate(rows, key, precision_target=0.9):
    pts = [(r["scores"][key]["overall"], r["iou"] >= IOU_GOOD) for r in rows
           if r["iou"] is not None and r["scores"][key]["overall"] is not None]
    pos = [s for s, g in pts if g]
    neg = [s for s, g in pts if not g]
    out = {"n": len(pts), "n_pos": len(pos), "n_neg": len(neg), "auc": auc(pos, neg),
           "tau": None}
    for tau in sorted({s for s, _ in pts}):                # lowest tau = most recall
        acc = [g for s, g in pts if s >= tau]
        prec = sum(acc) / len(acc)
        if prec >= precision_target:
            out.update(tau=tau, precision=round(prec, 4),
                       recall=round(sum(acc) / max(len(pos), 1), 4), accepted=len(acc))
            break
    return out


def make_sheet(best, path, tile_w=520):
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for ov, row in best:
        t = ov.resize((tile_w, int(ov.height * tile_w / ov.width)), Image.BICUBIC)
        d = ImageDraw.Draw(t)
        p, a = row["scores"]["primary"], row["scores"]["alt"]
        txt = (f"#{row.get('frame', '')} {row['kind']} p={row.get('person_p', '')}\n"
               f"O={fmt(p.get('overall'))} PF={fmt(p.get('prompt_following'))} "
               f"C={fmt(p.get('consistency'))} | alt O={fmt(a.get('overall'))}")
        d.rectangle((0, 0, tile_w, 46), fill=(0, 0, 0))
        d.multiline_text((5, 2), txt, fill=(255, 255, 0), font=font)
        tiles.append(t)
    cols, pad = 5, 4
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (th + pad) + pad), (40, 40, 40))
    for k, t in enumerate(tiles):
        sheet.paste(t, (pad + (k % cols) * (tile_w + pad), pad + (k // cols) * (th + pad)))
    sheet.save(path)
    print(f"sheet -> {path} {sheet.size}")


def fmt(v):
    return "NA" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coco")
    ap.add_argument("--images")
    ap.add_argument("--corrupt", action="store_true")
    ap.add_argument("--rfdetr")
    ap.add_argument("--frames")
    ap.add_argument("--frame-dir")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N images/frames only")
    ap.add_argument("--out", required=True, help="output prefix: <out>.jsonl, <out>.summary.json")
    ap.add_argument("--save-overlays", action="store_true", help="write <out>_overlays/*.png")
    ap.add_argument("--sheet", help="mode (b): sheet of the best-scored candidate per frame")
    ap.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    ap.add_argument("--num-pass", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--fast", action="store_true",
                    help="decider readout: score from logits at the forced score slot, no "
                         "reasoning; PQ once per mask; shared frame prefix, batched candidates")
    ap.add_argument("--fast-temperature", type=float, default=1.0)
    ap.add_argument("--compare", help="full-mode JSONL to validate --fast against (by image+kind)")
    ap.add_argument("--profile", help="profile every candidate of the first image, both "
                                      "instructions, and write the stage split here (json)")
    args = ap.parse_args()
    if bool(args.coco) == bool(args.rfdetr):
        ap.error("give exactly one of --coco or --rfdetr")

    gen = candidates_coco(args) if args.coco else candidates_rfdetr(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.save_overlays:
        os.makedirs(args.out + "_overlays", exist_ok=True)
    scorer = load_scorer(args.precision, args.num_pass)

    fast = FastScorer(scorer, args.fast_temperature) if args.fast else None
    rows, best, prof, prof_img = [], {}, [], None

    def groups():
        cur, key = [], None
        for src, m, row in gen:
            if cur and row["image"] != key:
                yield cur
                cur = []
            key = row["image"]
            cur.append((src, m, row))
        if cur:
            yield cur

    with open(args.out + ".jsonl", "w") as fh:
        for group in groups():
            src = group[0][0]
            ovs = [overlay(s, m) for s, m, _ in group]
            if args.profile and prof_img is None:
                prof_img = group[0][2]["image"]
                for (_, m, row), ov in zip(group, ovs):
                    for key, text in INSTRUCTIONS.items():
                        p = profile_evaluate(scorer, src, ov, text)
                        prof.append({"image": os.path.basename(row["image"]),
                                     "kind": row["kind"], "instruction": key, **p})
                        print("profile", row["kind"], key, json.dumps(p["share"]),
                              p["evaluate_total_s"], "s", flush=True)
                with open(args.profile, "w") as pf:
                    json.dump({"precision": args.precision, "max_pixels": args.max_pixels,
                               "gpu": __import__("torch").cuda.get_device_name(),
                               "evaluations": prof}, pf, indent=2)
            if fast:
                import torch
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.time()
                scored = fast.score_frame(src, ovs)
                torch.cuda.synchronize()
                per = round((time.time() - t0) / len(group), 3)
                peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                for sc in scored:
                    for key in INSTRUCTIONS:
                        sc[key]["seconds"] = per
                        sc[key]["peak_vram_gib"] = peak
                    sc["frame_seconds"] = round(per * len(group), 3)
            else:
                scored = [grade(scorer, src, ov) for ov in ovs]
            for (_, m, row), ov, sc in zip(group, ovs, scored):
                row["scores"] = sc
                row["mode"] = "fast" if fast else "full"
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                p = row["scores"]["primary"]
                print(f"{os.path.basename(row['image']):24} {row['kind']:16} iou={row['iou']} "
                      f"O={fmt(p['overall'])} altO={fmt(row['scores']['alt']['overall'])} "
                      f"{p['seconds']}s {p['peak_vram_gib']}GiB", flush=True)
                tag = f"{os.path.basename(row['image']).rsplit('.', 1)[0]}_" \
                      f"{row.get('instance', '')}{row['kind']}"
                if args.save_overlays:
                    ov.save(os.path.join(args.out + "_overlays", tag + ".png"))
                key = row["image"]
                score = p["overall"] if p["overall"] is not None else -1
                if key not in best or score > best[key][0]:
                    best[key] = (score, ov, row)

    timing = [r["scores"][k] for r in rows for k in INSTRUCTIONS]
    summary = {"mode": "coco" if args.coco else "rfdetr", "precision": args.precision,
               "max_pixels": args.max_pixels, "instructions": INSTRUCTIONS,
               "n_candidates": len(rows),
               "sec_per_eval_mean": round(float(np.mean([t["seconds"] for t in timing])), 2),
               "peak_vram_gib_max": max(t["peak_vram_gib"] for t in timing)}
    if args.coco:
        summary["calibration"] = {k: calibrate(rows, k) for k in INSTRUCTIONS}
        summary["iou_threshold"] = IOU_GOOD
    else:
        summary["best_per_frame"] = [
            {"image": os.path.basename(r["image"]), "kind": r["kind"],
             "person_p": r["person_p"], "overall": r["scores"]["primary"]["overall"],
             "overall_alt": r["scores"]["alt"]["overall"]}
            for _, _, r in best.values()]
    if args.compare:
        summary["vs_full"] = compare_full(rows, args.compare)
    with open(args.out + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "best_per_frame"}, indent=2))
    if args.sheet:
        make_sheet([(ov, r) for _, ov, r in best.values()], args.sheet)


if __name__ == "__main__":
    main()
