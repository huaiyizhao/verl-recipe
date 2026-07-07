#!/usr/bin/env python3
"""Micro-benchmark: does the vLLM<->HF(=FSDP training path) per-token logprob gap come from the
VISION/image path, or from a general logit-scale / attention / mrope difference?

Background: the RL run's rollout-correction RS mask fires because vLLM (rollout) and FSDP (training)
logprobs diverge. We already RULED OUT remove_padding packing (turning rmpad off left the gap
unchanged). The remaining question, from the LOGPROB_GAP dumps, is whether the gap is driven by
images (divergence concentrated on image-grounded natural-language tokens) or is just a high-entropy
logit-scale effect (temp_slope was large -> gap grows with token entropy, which images are confounded
with). This script separates the two by comparing vLLM vs HF on a controlled TEXT-ONLY prompt and a
TEXT+IMAGE prompt, with clean (freshly processed) pixels and the same tokenizer/chat-template.

Decision tree (compare the two summaries this prints):
  * TEXT gap ~0  AND  IMAGE gap large   -> ROOT is the VISION/image path (vLLM vision encoder vs
                                            transformers, or the RL pipeline feeding different pixels).
  * TEXT gap large (~ IMAGE gap)        -> ROOT is general vLLM-vs-HF numerics (attention backend /
                                            mrope / logit scale), NOT images. No vision fix helps.
  * IMAGE(microbench, clean pixels) << IMAGE(pipeline LOGPROB_GAP) -> the RL pipeline's pixel path
                                            (dedup / bf16 storage / transferqueue) adds corruption on
                                            top of the inherent gap. Chase the pixels (see --dump).

HF here is loaded exactly like verl's training forward (bf16 + flash_attention_2), so HF == the FSDP
numerics for this purpose (packing/sharding already shown irrelevant). vLLM is loaded FRESH (its own
clean image processing) so the microbench isolates the *inherent* vLLM-vs-HF gap from any pipeline
pixel corruption.

Run on the GPU box (needs vLLM + a GPU):
  python recipe/fully_async_gui_agent/logprob_vllm_vs_hf_microbench.py \
      --model /efs/data/models/Qwen3-VL-8B-Instruct --image /path/to/screenshot.png

  # text-only (no vision):        ... --model ...            (omit --image)
  # also check a dumped bundle's train-side pixels for bf16-rounding / stats:
  #                               ... --dump /efs/data/rl/logprob_probe_masked/probe_rank0_0.pt
"""

import argparse

import torch


def _p(msg):
    print(msg, flush=True)


def _summ(tag, vllm_lp, hf_lp, ids, tokenizer, topk=12):
    """Print a per-token table + summary of |vLLM - HF| over the generated tokens."""
    v = torch.tensor(vllm_lp, dtype=torch.float64)
    h = torch.tensor(hf_lp, dtype=torch.float64)
    n = min(len(v), len(h))
    v, h = v[:n], h[:n]
    d = v - h
    ad = d.abs()
    _p(f"\n===== [{tag}] vLLM vs HF over {n} generated tokens =====")
    _p(
        f"[{tag}] mean_abs_d={ad.mean():.5f}  max_abs_d={ad.max():.5f}  "
        f"mean_signed_d={d.mean():+.5f}  frac|d|>0.1={(ad > 0.1).float().mean():.3f}  "
        f"frac|d|>0.5={(ad > 0.5).float().mean():.3f}"
    )
    # WHERE does the |d| mass concentrate?
    # (a) direction split — is the gap systematic (vLLM over/under-confident) or symmetric noise?
    tot = ad.sum().clamp(min=1e-9)
    _p(
        f"[{tag}] direction: |d| mass  vLLM-higher(d>0)={ad[d > 0].sum() / tot * 100:4.0f}%  "
        f"vLLM-lower(d<0)={ad[d < 0].sum() / tot * 100:4.0f}%"
    )
    # (b) by vLLM's own confidence — does the gap live on near-deterministic tokens or high-entropy ones?
    #     (high-entropy = the model is choosing among alternatives = where vision grounding matters).
    for name, m in (
        ("confident vLLM_lp>-0.5 ", v > -0.5),
        ("mid  -2<=lp<=-0.5      ", (v <= -0.5) & (v >= -2.0)),
        ("uncertain  lp<-2       ", v < -2.0),
    ):
        c = int(m.sum())
        if c:
            _p(
                f"[{tag}]   {name}: n={c:3d}  mean_abs_d={ad[m].mean():.4f}  "
                f"mass={ad[m].sum() / tot * 100:4.0f}%  mean_signed={d[m].mean():+.4f}"
            )
    # worst tokens
    order = torch.argsort(ad, descending=True)[:topk]
    _p(f"[{tag}] worst {topk} tokens (pos | vLLM  HF  d | token):")
    for i in order.tolist():
        tok = tokenizer.decode([int(ids[i])]) if tokenizer is not None else str(int(ids[i]))
        _p(f"    {i:4d} | {v[i]:+8.3f} {h[i]:+8.3f} {d[i]:+7.3f} | {tok!r}")
    return float(ad.mean()), float(ad.max())


def _pixel_stats(tag, pv):
    """Report pixel_values integrity: dtype/shape/range + whether it is bf16-rounded (a fingerprint of
    the RL pipeline having stored images in bf16 -> train side loses precision vs vLLM's fp32)."""
    pvf = pv.float()
    bf16_rt = torch.equal(pvf, pvf.to(torch.bfloat16).to(torch.float32))
    _p(
        f"[PIXELS {tag}] shape={tuple(pv.shape)} dtype={pv.dtype} "
        f"min={pvf.min():.4f} max={pvf.max():.4f} mean={pvf.mean():.4f} std={pvf.std():.4f} "
        f"nan={bool(torch.isnan(pvf).any())} | bf16_roundtrip_equal={bf16_rt} "
        f"{'<- pixels are bf16-quantized (train/infer precision mismatch!)' if bf16_rt else '<- genuine fp32'}"
    )


def _run_async_bundle(path, hf, processor, tokenizer, device):
    """Load an async-run probe bundle and 3-way compare, per response token:
        A = async's recorded old_log_probs  (vLLM, since bypass_mode sets old = rollout_log_probs)
        B = async's recorded fsdp_log_probs (the async FSDP forward during that run)
        C = a FRESH HF/FSDP recompute here, on the SAME tokens+pixels.
    |A-B| reproduces async's tiny recorded gap. |A-C| is the KEY: if small, async's "vLLM" old_log_prob
    actually matches a fresh FSDP forward on real GUI data (=> the real vLLM<->FSDP gap is small, and the
    microbench's 0.28 was a verbose-prompt artifact); if ~0.28, async old IS vLLM-with-vision-gap.
    |B-C| is a sanity check (async FSDP vs fresh HF; should be ~bf16 noise)."""
    _p(f"=== [ASYNC-BUNDLE] loading {path} ===")
    b = torch.load(path, map_location="cpu", weights_only=False)

    def _nested(field):
        f = b.get(field)
        if f is None:
            return None, None
        if isinstance(f, dict) and "values" in f:
            return f["values"], f["offsets"].tolist()
        return f, None

    ids_v, ids_off = _nested("input_ids")
    old_v, old_off = _nested("old_log_probs")  # A
    fsdp_v, fsdp_off = _nested("fsdp_log_probs")  # B
    rm_v, rm_off = _nested("response_mask")
    pix, _ = _nested("pixel_values")
    grid, _ = _nested("image_grid_thw")
    if ids_off is None:
        _p("[ASYNC-BUNDLE] input_ids not nested per-seq; cannot proceed")
        return
    n_seq = len(ids_off) - 1
    image_token_id = getattr(processor, "image_token_id", None)
    merge = processor.image_processor.merge_size
    _p(f"[ASYNC-BUNDLE] n_seq={n_seq} has_A(old/vLLM)={old_v is not None} has_B(fsdp)={fsdp_v is not None} "
       f"has_pixels={pix is not None}")

    img_cursor, patch_cursor = 0, 0
    accA, accB, accC = [], [], []
    for i in range(n_seq):
        ids_i = ids_v[ids_off[i]:ids_off[i + 1]].to(device)
        L = ids_i.numel()
        A = old_v[old_off[i]:old_off[i + 1]].float() if old_v is not None else None
        Bfull = fsdp_v[fsdp_off[i]:fsdp_off[i + 1]].float() if fsdp_v is not None else None
        rm = rm_v[rm_off[i]:rm_off[i + 1]] if rm_v is not None else None
        resp_len = A.numel() if A is not None else (rm.numel() if rm is not None else 0)
        if resp_len == 0:
            continue
        B = Bfull[-resp_len:] if Bfull is not None else None  # response tail of the full FSDP logp
        # split pixels for this seq (images appear in order; consume by patch count)
        n_img_tok = int((ids_i.cpu() == image_token_id).sum()) if image_token_id is not None else 0
        seq_grids, seq_pix, consumed = [], [], 0
        while grid is not None and img_cursor < grid.shape[0] and consumed < n_img_tok:
            g = grid[img_cursor]
            patches = int(g[0] * g[1] * g[2])
            seq_grids.append(g)
            if pix is not None:
                seq_pix.append(pix[patch_cursor:patch_cursor + patches])
            patch_cursor += patches
            consumed += patches // (merge * merge)
            img_cursor += 1
        grid_i = torch.stack(seq_grids).to(device) if seq_grids else None
        pix_i = torch.cat(seq_pix).to(device) if seq_pix else None
        # fresh HF forward -> logp of each response token (tail alignment)
        with torch.no_grad():
            kw = {"input_ids": ids_i.unsqueeze(0), "use_cache": False}
            if pix_i is not None:
                kw["pixel_values"] = pix_i.to(hf.dtype)
                kw["image_grid_thw"] = grid_i
            lp = torch.log_softmax(hf(**kw).logits[0].float(), dim=-1)
        C = torch.tensor([float(lp[L - resp_len + j - 1, int(ids_i[L - resp_len + j])]) for j in range(resp_len)])
        m = rm.bool() if (rm is not None and rm.numel() == resp_len) else torch.ones(resp_len, dtype=torch.bool)
        Cm = C[m]
        Am = A[m] if A is not None else None
        Bm = B[m] if B is not None else None
        if Am is not None:
            accA.append(Am)
        if Bm is not None:
            accB.append(Bm)
        accC.append(Cm)
        dab = (Am - Bm).abs().mean().item() if (Am is not None and Bm is not None) else float("nan")
        dac = (Am - Cm).abs().mean().item() if Am is not None else float("nan")
        dbc = (Bm - Cm).abs().mean().item() if Bm is not None else float("nan")
        _p(f"[ASYNC-BUNDLE] seq{i} L={L} resp={resp_len} imgtok={n_img_tok} | "
           f"|A-B|async_gap={dab:.4f} |A-C|asyncVLLM_vs_freshFSDP={dac:.4f} |B-C|asyncFSDP_vs_freshFSDP={dbc:.4f}")

    A = torch.cat(accA) if accA else None
    B = torch.cat(accB) if accB else None
    C = torch.cat(accC)
    _p("\n[ASYNC-BUNDLE] ===== AGGREGATE over all response tokens =====")
    if A is not None and B is not None:
        d = (A - B).abs()
        _p(f"  |A-B| async recorded gap (vLLM vs FSDP): mean={d.mean():.4f} max={d.max():.4f}")
    if A is not None:
        d = (A - C).abs()
        _p(f"  |A-C| async-vLLM(old) vs FRESH-HF/FSDP : mean={d.mean():.4f} max={d.max():.4f}")
    if B is not None:
        d = (B - C).abs()
        _p(f"  |B-C| async-FSDP vs FRESH-HF (sanity)  : mean={d.mean():.4f} max={d.max():.4f}")
    _p("  READ: |A-C| small (~|B-C|) => async 'vLLM' old_log_prob matches a fresh FSDP forward => the real")
    _p("        vLLM<->FSDP gap on GUI data is SMALL (microbench's 0.28 was the verbose-prompt artifact).")
    _p("        |A-C| ~0.28 while |B-C| small => async old IS vLLM-with-vision-gap; async's small recorded")
    _p("        |A-B| would then be impossible -> so this case points to old_log_prob NOT being raw vLLM.")


def _run_rollout_bundle(path, model_path, hf, processor, tokenizer, device, gpu_mem):
    """Load a ROLLOUT probe (raw images + tokens + recorded vLLM logprobs) and 3-way compare per response
    token: A=recorded-vLLM (as the async run stored it), C=fresh-HF, D=fresh-vLLM (recomputed here on the
    SAME raw images + tokens). |A-D| = is the recorded vLLM logprob reproducible; |C-D| = clean vLLM<->FSDP
    gap on real GUI data; |A-C| = recorded-vLLM vs fresh-FSDP."""
    _p(f"=== [ROLLOUT-BUNDLE] loading {path} ===")
    b = torch.load(path, map_location="cpu", weights_only=False)
    prompt_ids = [int(x) for x in b["prompt_ids"]]
    response_ids = [int(x) for x in b["response_ids"]]
    A = torch.tensor([float(x) for x in b["response_logprobs"]], dtype=torch.float64)  # recorded vLLM
    mm = b.get("multi_modal_data") or {}
    images = mm.get("images") or mm.get("image") or []
    full = prompt_ids + response_ids
    plen, rlen = len(prompt_ids), len(response_ids)
    n = min(rlen, A.numel())
    _p(f"[ROLLOUT-BUNDLE] prompt={plen} response={rlen} n_images={len(images)} recorded_vLLM_logp_n={A.numel()}")

    # ---- C: fresh HF/FSDP on the same raw images + tokens ----
    C = None
    try:
        img_inputs = processor.image_processor(images, return_tensors="pt") if images else {}
        pv = img_inputs.get("pixel_values")
        grid = img_inputs.get("image_grid_thw")
        with torch.no_grad():
            kw = {"input_ids": torch.tensor(full, device=device).unsqueeze(0), "use_cache": False}
            if pv is not None:
                kw["pixel_values"] = pv.to(hf.dtype).to(device)
                kw["image_grid_thw"] = grid.to(device)
            lp = torch.log_softmax(hf(**kw).logits[0].float(), dim=-1)
        C = torch.tensor([float(lp[plen + j - 1, full[plen + j]]) for j in range(n)], dtype=torch.float64)
    except Exception as e:  # noqa: BLE001
        _p(f"[ROLLOUT-BUNDLE] fresh-HF (C) failed: {e!r}")

    # ---- D: fresh vLLM on the same raw images + tokens (teacher-forcing via prompt_logprobs) ----
    D = None
    try:
        from vllm import LLM, SamplingParams

        _p("[ROLLOUT-BUNDLE] loading vLLM for fresh recompute (D)...")
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max(1, len(images))},
            enforce_eager=True,
        )
        sp = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0)
        req = {"prompt_token_ids": full}
        if images:
            req["multi_modal_data"] = {"image": images}
        out = llm.generate([req], sp)[0]
        pl = out.prompt_logprobs  # list aligned with `full`; pl[i] = {token_id: Logprob} for position i
        Dvals = []
        for j in range(n):
            p = plen + j
            entry = pl[p] if pl is not None and p < len(pl) else None
            tok = full[p]
            Dvals.append(entry[tok].logprob if (entry and tok in entry) else float("nan"))
        D = torch.tensor(Dvals, dtype=torch.float64)
    except Exception as e:  # noqa: BLE001
        _p(f"[ROLLOUT-BUNDLE] fresh-vLLM (D) failed ({e!r}); reporting A vs C only")

    A = A[:n]

    def _cmp(name, x, y):
        d = (x - y).abs()
        _p(f"  {name}: mean={d.mean():.4f} max={d.max():.4f} frac>0.1={(d > 0.1).float().mean():.3f}")

    _p("\n[ROLLOUT-BUNDLE] ===== per-response-token comparison =====")
    if D is not None:
        _cmp("|A-D| recorded-vLLM vs FRESH-vLLM  ", A, D)
    if C is not None:
        _cmp("|A-C| recorded-vLLM vs FRESH-HF/FSDP", A, C)
    if C is not None and D is not None:
        _cmp("|C-D| FRESH-HF vs FRESH-vLLM (clean gap)", C, D)
    # worst tokens on the clean gap
    if C is not None and D is not None:
        order = torch.argsort((C - D).abs(), descending=True)[:12]
        _p("  worst |C-D| tokens (pos | A  C  D | token):")
        for i in order.tolist():
            tok = tokenizer.decode([full[plen + i]])
            _p(f"    {i:4d} | {A[i]:+7.3f} {C[i]:+7.3f} {D[i]:+7.3f} | {tok!r}")
    _p("  READ: |A-D| small => recorded vLLM is a faithful vLLM recompute (async old IS vLLM).")
    _p("        |C-D| is the REAL vLLM<->FSDP gap on this GUI data; compare it to async's recorded |A-B|.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None, help="Path to a screenshot; enables the TEXT+IMAGE test.")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Match training (1.0). At 1.0 vLLM's returned logprobs are raw (no temp scaling), so they "
        "compare apples-to-apples with HF's log_softmax.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2", help="HF attn_implementation (match verl).")
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM gpu_memory_utilization.")
    ap.add_argument("--dump", default=None, help="Optional probe .pt bundle -> report its train-side pixel stats.")
    ap.add_argument(
        "--async-bundle",
        dest="async_bundle",
        default=None,
        help="A probe .pt dumped from the ASYNC run (tokens+pixels+old_log_probs[vLLM]+fsdp_log_probs[FSDP]). "
        "Recompute a FRESH HF/FSDP logprob on the SAME data and 3-way compare A=async-vLLM(old), "
        "B=async-FSDP(fsdp), C=fresh-HF. Answers: is the async old_log_prob vLLM-with-gap or FSDP-consistent, "
        "and what is the real vLLM<->FSDP gap on actual GUI data. Runs HF only (no vLLM), then exits.",
    )
    ap.add_argument(
        "--text-prompt",
        default="Write a detailed, imaginative short story about a lighthouse keeper who discovers "
        "something impossible washed up on the shore one foggy morning. Be creative and specific.",
        help="A high-entropy open-ended text prompt (no image).",
    )
    ap.add_argument(
        "--image-prompt",
        default="You are a GUI agent. Look at this screenshot and describe, step by step and in "
        "specific detail, what is on the screen and what you would click to open the main menu.",
    )
    ap.add_argument(
        "--rollout-bundle",
        dest="rollout_bundle",
        default=None,
        help="A ROLLOUT probe .pt (raw images + tokens + recorded vLLM logprobs). Recompute a FRESH vLLM (D) "
        "and FRESH HF (C) on the SAME raw images/tokens and 3-way compare to A=recorded-vLLM. |A-D| tells "
        "whether the recorded 'vLLM' logprob is a faithful vLLM recompute; |C-D| is the clean vLLM<->FSDP gap.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- offline pixel integrity check on a dumped bundle (train side) -------------------------
    if args.dump:
        _p(f"=== [DUMP] loading {args.dump} for train-side pixel integrity ===")
        b = torch.load(args.dump, map_location="cpu", weights_only=False)
        pv = b.get("pixel_values")
        grid = b.get("image_grid_thw")
        if pv is None:
            _p("[DUMP] no pixel_values in bundle")
        else:
            _pixel_stats("dump-train-side", pv if isinstance(pv, torch.Tensor) else pv.values())
            if grid is not None:
                g = grid if isinstance(grid, torch.Tensor) else grid.values()
                _p(f"[DUMP] image_grid_thw={tuple(g.shape)} sum(t*h*w)={int((g[:, 0] * g[:, 1] * g[:, 2]).sum())}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    _p(f"=== loading HF model (bf16, attn={args.attn}) — this IS the FSDP training-path numerics ===")
    hf = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn, trust_remote_code=True
    ).eval().to(device)

    # ---- BUNDLE MODES: run BOTH in one shot when given, then exit (no generation needed) -------------
    #   --async-bundle   : A=async-vLLM(old) / B=async-FSDP(fsdp) / C=fresh-HF   (HF only)
    #   --rollout-bundle : A=recorded-vLLM   / C=fresh-HF        / D=fresh-vLLM  (loads vLLM)
    if args.async_bundle or args.rollout_bundle:
        if args.async_bundle:
            _p("\n################## ASYNC-BUNDLE (A=async vLLM / B=async FSDP / C=fresh HF) ##################")
            _run_async_bundle(args.async_bundle, hf, processor, tokenizer, device)
        if args.rollout_bundle:
            _p("\n################## ROLLOUT-BUNDLE (A=recorded vLLM / C=fresh HF / D=fresh vLLM) ##################")
            _run_rollout_bundle(args.rollout_bundle, args.model, hf, processor, tokenizer, device, args.gpu_mem)
        return

    from vllm import LLM, SamplingParams

    _p("=== loading vLLM (fresh, its own clean image processing) ===")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 4},
        enforce_eager=True,  # determinism: skip cudagraph
    )
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.n_tokens, logprobs=1, seed=args.seed)

    def hf_logp_for(full_ids, start, pixel_values=None, image_grid_thw=None):
        """Teacher-force full_ids through HF; return logp of each token at positions [start, len).
        logits[t-1] predicts token t -> logp for token at absolute pos t (t from `start`)."""
        with torch.no_grad():
            kw = {"input_ids": full_ids.unsqueeze(0).to(device), "use_cache": False}
            if pixel_values is not None:
                kw["pixel_values"] = pixel_values.to(hf.dtype).to(device)
                kw["image_grid_thw"] = image_grid_thw.to(device)
            logits = hf(**kw).logits[0].float()  # (L, V); let HF build mrope from ids+grid internally
            lp = torch.log_softmax(logits, dim=-1)
            out = []
            for t in range(start, full_ids.numel()):
                out.append(float(lp[t - 1, int(full_ids[t])]))
            return out

    results = {}

    # ---------- TEXT-ONLY ----------
    _p("\n########## TEXT-ONLY ##########")
    msgs = [{"role": "user", "content": args.text_prompt}]
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    out = llm.generate([{"prompt_token_ids": prompt_ids.tolist()}], sp)[0].outputs[0]
    gen_ids = list(out.token_ids)
    vllm_lp = [out.logprobs[i][gen_ids[i]].logprob for i in range(len(gen_ids))]
    full = torch.cat([prompt_ids, torch.tensor(gen_ids, dtype=prompt_ids.dtype)])
    hf_lp = hf_logp_for(full, start=prompt_ids.numel())
    _p(f"[TEXT] response: {tokenizer.decode(gen_ids)[:300]!r}")
    results["TEXT"] = _summ("TEXT", vllm_lp, hf_lp, gen_ids, tokenizer)

    # ---------- TEXT+IMAGE ----------
    if args.image:
        from PIL import Image

        _p("\n########## TEXT+IMAGE ##########")
        img = Image.open(args.image).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.image_prompt}]}]
        prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # HF-side processed inputs (this is the CLEAN pixel reference)
        proc = processor(text=[prompt_text], images=[img], return_tensors="pt")
        prompt_ids = proc["input_ids"][0]
        pixel_values = proc["pixel_values"]
        image_grid_thw = proc["image_grid_thw"]
        _pixel_stats("microbench-clean", pixel_values)
        # vLLM generate on the same raw image (keep the full RequestOutput to read vLLM's own prompt)
        vout = llm.generate([{"prompt": prompt_text, "multi_modal_data": {"image": img}}], sp)[0]
        out = vout.outputs[0]

        # PREPROCESSING vs ENCODER discriminator: compare how many image tokens each side expanded the
        # picture into. HF uses `processor`; vLLM does its OWN preprocessing (smart_resize/patchify). If
        # the counts differ, vLLM fed the vision tower a DIFFERENT-resolution image -> the gap is (at
        # least partly) PREPROCESSING and is fixable by aligning the mm config (min/max pixels). If the
        # counts MATCH but the logp gap stays, pixels agree -> the gap is the vision ENCODER numerics.
        img_tok_id = getattr(processor, "image_token_id", None)
        if img_tok_id is None:
            img_tok_id = getattr(hf.config, "image_token_id", None)
        n_img_hf = int((prompt_ids == img_tok_id).sum()) if img_tok_id is not None else -1
        vllm_prompt_ids = list(vout.prompt_token_ids) if getattr(vout, "prompt_token_ids", None) else []
        n_img_vllm = sum(1 for t in vllm_prompt_ids if t == img_tok_id) if img_tok_id is not None else -1
        _p(
            f"[IMG-TOKENS] HF processor={n_img_hf}  vLLM={n_img_vllm}  match={n_img_hf == n_img_vllm}  "
            f"| pixel_values patches={pixel_values.shape[0]} (n_tokens=patches/merge^2); "
            f"vLLM_prompt_len={len(vllm_prompt_ids)} HF_prompt_len={prompt_ids.numel()}  "
            f"{'<- MISMATCH => vLLM preprocessing differs (resolution/resize)' if n_img_hf != n_img_vllm else '<- counts match => preprocessing same, gap is encoder numerics'}"
        )

        gen_ids = list(out.token_ids)
        vllm_lp = [out.logprobs[i][gen_ids[i]].logprob for i in range(len(gen_ids))]
        full = torch.cat([prompt_ids, torch.tensor(gen_ids, dtype=prompt_ids.dtype)])
        hf_lp = hf_logp_for(full, start=prompt_ids.numel(), pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        _p(f"[IMAGE] response: {tokenizer.decode(gen_ids)[:300]!r}")
        results["IMAGE"] = _summ("IMAGE", vllm_lp, hf_lp, gen_ids, tokenizer)

        # ---------- fast_pos_embed_interpolate PATCH A/B ----------
        # verl-v1 monkey-patches the vision pos-embed interpolation onto Qwen3-VL; verl-async does NOT.
        # v1's train<->infer LOGPROB_GAP is ~0.28, async's is ~0.01. Test HERE whether that patch is the
        # cause: (1) diff the STOCK method (what this HF model uses, == verl-async) vs verl's CUSTOM one;
        # (2) if they differ, re-run the HF image logprob WITH the custom patch and see if the vLLM gap
        # jumps toward v1's 0.28. IDENTICAL => patch inert, NOT the cause. DIFFERENT + gap jumps => cause.
        try:
            vision = getattr(hf, "visual", None) or getattr(getattr(hf, "model", None), "visual", None)
            if vision is None or not hasattr(vision, "fast_pos_embed_interpolate"):
                _p("[POSEMB] vision.fast_pos_embed_interpolate not found; A/B skipped")
            else:
                # import verl's custom impl (defensive: it lives in qwen3_vl.py or qwen3_5.py)
                custom_fn = None
                for mod in ("verl.models.transformers.qwen3_vl", "verl.models.transformers.qwen3_5"):
                    try:
                        custom_fn = __import__(mod, fromlist=["fast_pos_embed_interpolate"]).fast_pos_embed_interpolate
                        _p(f"[POSEMB] loaded verl custom fast_pos_embed_interpolate from {mod}")
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if custom_fn is None:
                    _p("[POSEMB] could not import verl custom fast_pos_embed_interpolate; A/B skipped")
                else:
                    g = image_grid_thw.to(device)
                    with torch.no_grad():
                        stock_pe = vision.fast_pos_embed_interpolate(g).float()
                        custom_pe = custom_fn(vision, g).float()
                    pe_d = (stock_pe - custom_pe).abs()
                    _p(
                        f"[POSEMB] stock(=verl-async) vs verl-custom(=v1) output: "
                        f"max_abs_d={pe_d.max().item():.6f} mean_abs_d={pe_d.mean().item():.6f} "
                        f"shape={tuple(stock_pe.shape)}  "
                        f"{'<- IDENTICAL => patch INERT, NOT the cause' if pe_d.max().item() < 1e-5 else '<- DIFFERENT => patch changes vision embeds'}"
                    )
                    # end-to-end: rerun HF image logprob with the custom patch installed on the class
                    cls = type(vision)
                    orig = cls.fast_pos_embed_interpolate
                    cls.fast_pos_embed_interpolate = custom_fn
                    try:
                        hf_lp_custom = hf_logp_for(
                            full, start=prompt_ids.numel(), pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                    finally:
                        cls.fast_pos_embed_interpolate = orig
                    _p("[POSEMB] re-running IMAGE gap with the verl-v1 CUSTOM pos-embed patch applied:")
                    _summ("IMAGE+v1PosembPatch", vllm_lp, hf_lp_custom, gen_ids, tokenizer)
        except Exception as e:  # noqa: BLE001
            _p(f"[POSEMB] A/B failed ({e!r})")

    # ---------- VERDICT ----------
    _p("\n########## VERDICT ##########")
    tmean, tmax = results["TEXT"]
    _p(f"TEXT : mean_abs_d={tmean:.4f} max_abs_d={tmax:.4f}")
    if "IMAGE" in results:
        imean, imax = results["IMAGE"]
        _p(f"IMAGE: mean_abs_d={imean:.4f} max_abs_d={imax:.4f}")
        if tmean < 0.1 and imean > 3 * max(tmean, 1e-6):
            _p(">> VISION path: text agrees, image diverges -> the gap is driven by the image/vision forward.")
        elif imean < 2 * max(tmean, 1e-6):
            _p(">> GENERAL numerics: image ~ text -> attention backend / mrope / logit-scale, NOT vision.")
        else:
            _p(">> Inconclusive: compare magnitudes above; also compare IMAGE(here) vs the pipeline LOGPROB_GAP.")
    else:
        _p("(no --image; run again with --image to get the vision-vs-text split.)")


if __name__ == "__main__":
    main()
