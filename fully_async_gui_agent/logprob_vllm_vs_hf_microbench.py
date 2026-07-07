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


def _k3(ref_logp, other_logp):
    """Per-token k3 approximate KL with tokens sampled from ref: exp(other-ref)-1-(other-ref)."""
    n = min(ref_logp.numel(), other_logp.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.float64)
    ref = ref_logp[:n].to(torch.float64)
    other = other_logp[:n].to(torch.float64)
    d = (other - ref).clamp(-20, 20)
    out = torch.exp(d) - 1.0 - d
    return out[torch.isfinite(out)]


def _k3_mean(ref_logp, other_logp):
    v = _k3(ref_logp, other_logp)
    return float(v.mean()) if v.numel() else float("nan")


def _fmt(v):
    return "nan" if not torch.isfinite(torch.tensor(v)) else f"{v:.5f}"


def _print_k3_matrix(tag, seq_rows):
    """Print seq-level pairwise k3 over whatever named logprob streams are available.

    Label convention: K3(X||Y) means tokens are sampled from X and scored by Y, matching the
    rollout-correction code path where d = logp_Y - logp_X.
    """
    pairs = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
    have_any = False
    _p(f"\n[{tag}] ===== seq-level k3 approximate KL =====")
    _p(f"[{tag}] convention: K3(X||Y)=mean(exp(logp_Y-logp_X)-1-(logp_Y-logp_X)) over response tokens")
    for row in seq_rows:
        vals = []
        for a, b in pairs:
            if row.get(a) is not None and row.get(b) is not None:
                vals.append(f"K3({a}||{b})={_fmt(_k3_mean(row[a], row[b]))}")
        if vals:
            have_any = True
            _p(f"[{tag}] seq{row['seq']} n={row['n']} " + " ".join(vals))
    agg = {}
    for a, b in pairs:
        xs, ys = [], []
        for row in seq_rows:
            if row.get(a) is not None and row.get(b) is not None:
                n = min(row[a].numel(), row[b].numel())
                if n:
                    xs.append(row[a][:n])
                    ys.append(row[b][:n])
        if xs:
            agg[(a, b)] = _k3_mean(torch.cat(xs), torch.cat(ys))
    if agg:
        _p(f"[{tag}] aggregate " + " ".join(f"K3({a}||{b})={_fmt(v)}" for (a, b), v in agg.items()))
    if not have_any:
        _p(f"[{tag}] no pairwise streams available for k3")


def _run_async_bundle(path, hf, processor, tokenizer, device, feed_pos=True):
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

    # Fields can be nested ({values, offsets}) OR plain padded tensors (n_seq, ...). Print + handle both.
    for k in ("input_ids", "position_ids", "old_log_probs", "fsdp_log_probs", "response_mask", "pixel_values",
              "image_grid_thw"):
        v = b.get(k)
        if isinstance(v, dict) and "values" in v:
            _p(f"  {k}: nested values={tuple(v['values'].shape)} n_off={len(v['offsets'])}")
        elif torch.is_tensor(v):
            _p(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            _p(f"  {k}: {type(v).__name__}")

    ii = b.get("input_ids")
    if isinstance(ii, dict) and "values" in ii:
        n_seq = len(ii["offsets"]) - 1
    elif torch.is_tensor(ii):
        n_seq = ii.shape[0] if ii.dim() >= 2 else 1
    else:
        _p("[ASYNC-BUNDLE] no usable input_ids; cannot proceed")
        return

    def _seq(name, i):
        v = b.get(name)
        if v is None:
            return None
        if isinstance(v, dict) and "values" in v:
            off = v["offsets"].tolist()
            return v["values"][off[i]:off[i + 1]]
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == n_seq:
            return v[i]
        if torch.is_tensor(v) and v.dim() == 1 and n_seq == 1:
            return v
        return None

    def _seq_pos(i):
        # mrope position_ids: (channels[,batch], seq). Jagged on the LAST dim. Return (channels, seq_len)
        # for seq i, so we can feed the EXACT training position_ids to HF (MRoPE is sensitive to these).
        v = b.get("position_ids")
        if v is None:
            return None
        if isinstance(v, dict) and "values" in v:
            vv, off = v["values"], v["offsets"].tolist()
            if vv.dim() == 2 and vv.shape[0] in (3, 4):  # (channels, total)
                return vv[:, off[i]:off[i + 1]]
            if vv.dim() == 3 and vv.shape[0] in (3, 4):  # (channels, 1, total)
                return vv[:, 0, off[i]:off[i + 1]]
            return vv[off[i]:off[i + 1]]  # (total,) fallback
        if torch.is_tensor(v):  # plain: (channels, n_seq, seq) or (n_seq, channels, seq)
            if v.dim() == 3 and v.shape[0] in (3, 4):
                return v[:, i]
            if v.dim() == 3 and v.shape[1] in (3, 4):
                return v[i]
        return None

    pix = b.get("pixel_values")
    grid = b.get("image_grid_thw")
    image_token_id = getattr(processor, "image_token_id", None)
    merge = processor.image_processor.merge_size
    _p(f"[ASYNC-BUNDLE] n_seq={n_seq}")

    img_cursor, patch_cursor = 0, 0
    accA, accB, accC = [], [], []
    seq_rows = []
    for i in range(n_seq):
        ids_i = _seq("input_ids", i)
        if ids_i is None:
            continue
        ids_i = ids_i.to(device).long().reshape(-1)
        L = ids_i.numel()
        A = _seq("old_log_probs", i)
        Bfull = _seq("fsdp_log_probs", i)
        rm = _seq("response_mask", i)
        A = A.float().reshape(-1) if A is not None else None
        Bfull = Bfull.float().reshape(-1) if Bfull is not None else None
        rm = rm.reshape(-1).bool() if rm is not None else None
        # valid response length (drop right-padding via response_mask)
        resp_valid = int(rm.sum()) if rm is not None else (A.numel() if A is not None else 0)
        if resp_valid == 0 or resp_valid > L:
            continue
        # A (padded response logp) -> valid entries
        Av = (A[rm] if (A is not None and rm is not None and rm.numel() == A.numel()) else (A[:resp_valid] if A is not None else None))
        # B: full-seq logp. verl uses the ROLLED convention (log_probs[t] = logp of input_ids[t+1]), so the
        # response tokens at abs pos [L-resp_valid, L-1] have their logp at indices [L-resp_valid-1, L-2].
        if Bfull is not None:
            if Bfull.numel() == L:
                Bv = Bfull[L - resp_valid - 1:L - 1]
            elif rm is not None and rm.numel() == Bfull.numel():
                Bv = Bfull[rm]
            else:
                Bv = Bfull[-resp_valid:]
        else:
            Bv = None
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
        # fresh HF forward -> logp of the last `resp_valid` (response) tokens
        with torch.no_grad():
            kw = {"input_ids": ids_i.unsqueeze(0), "use_cache": False}
            if pix_i is not None:
                kw["pixel_values"] = pix_i.to(hf.dtype)
                kw["image_grid_thw"] = grid_i
            # feed the EXACT training position_ids (MRoPE) so C uses the same positions as training FSDP,
            # not HF's internally-recomputed ones. Shape -> (channels, batch=1, seq_len).
            # --no-position-ids sets feed_pos=False to isolate: if |B-C| jumps back to ~0.28, position_ids
            # (not the monkey_patch) are what closed the gap.
            pos_i = _seq_pos(i) if feed_pos else None
            if pos_i is not None:
                pos_i = pos_i.to(device).long()
                if pos_i.dim() == 2 and pos_i.shape[0] in (3, 4) and pos_i.shape[1] == L:
                    kw["position_ids"] = pos_i.unsqueeze(1)  # (channels, 1, L)
                elif i == 0:
                    _p(f"[ASYNC-BUNDLE] position_ids shape {tuple(pos_i.shape)} != (channels,{L}); letting HF recompute")
            elif i == 0:
                _p(f"[ASYNC-BUNDLE] feed_pos={feed_pos}: {'letting HF recompute MRoPE positions' if not feed_pos else 'no position_ids in dump'}")
            lp = torch.log_softmax(hf(**kw).logits[0].float(), dim=-1)
        Cm = torch.tensor(
            [float(lp[L - resp_valid + j - 1, int(ids_i[L - resp_valid + j])]) for j in range(resp_valid)]
        )
        # defensive: align all to the common valid length
        k = min(x.numel() for x in (Cm,) + ((Av,) if Av is not None else ()) + ((Bv,) if Bv is not None else ()))
        Cm = Cm[:k]
        Am = Av[:k] if Av is not None else None
        Bm = Bv[:k] if Bv is not None else None
        if Am is not None:
            accA.append(Am)
        if Bm is not None:
            accB.append(Bm)
        accC.append(Cm)
        seq_rows.append(
            {
                "seq": i,
                "n": k,
                "A": Am.detach().cpu() if Am is not None else None,
                "B": Bm.detach().cpu() if Bm is not None else None,
                "C": Cm.detach().cpu(),
                "D": None,
            }
        )
        dab = (Am - Bm).abs().mean().item() if (Am is not None and Bm is not None) else float("nan")
        dac = (Am - Cm).abs().mean().item() if Am is not None else float("nan")
        dbc = (Bm - Cm).abs().mean().item() if Bm is not None else float("nan")
        _p(f"[ASYNC-BUNDLE] seq{i} L={L} resp_valid={resp_valid} imgtok={n_img_tok} | "
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
    _print_k3_matrix("ASYNC-BUNDLE", seq_rows)
    _p("[ASYNC-BUNDLE] D=fresh-vLLM is not available from this bundle: it stores processed pixel_values, "
       "not raw images. Use --rollout-bundle or dump raw images to compare D against A.")
    _p("  READ: with verl monkey_patch + the EXACT training position_ids fed, C reproduces training FSDP.")
    _p("        |B-C| ~0 => microbench == training FSDP; |A-C| ~0 => training vLLM == FSDP == microbench.")
    _p("        A ~0.28 gap only appears vs STOCK HF (no verl patch / wrong MRoPE positions) -> that was")
    _p("        the artifact, NOT a real train-infer gap. If |B-C| is still large, the patch/positions")
    _p("        aren't being applied (check the [VERL-PATCH] line and the position_ids shape print).")


def _run_prove_positions(path, hf, processor, tokenizer, device):
    """PROOF that v1's train<->infer gap is MRoPE position_ids computed WITHOUT mm_token_type_ids.
    For each dumped seq: recompute position_ids the ASYNC way (verl.utils.model.compute_vlm_position_ids,
    which builds mm_token_type_ids) and the V1 way (agent_loop._compute_position_ids logic: get_rope_index
    WITHOUT mm_token_type_ids), compare BOTH to the DUMPED ground-truth positions (what training used, which
    matched vLLM at 0.01). Then feed each to HF and print |A-C|. Expect: async-way == dump (|A-C|~0.006),
    v1-way != dump (|A-C|~0.28)."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    ii = b.get("input_ids")
    n_seq = len(ii["offsets"]) - 1 if isinstance(ii, dict) else (ii.shape[0] if torch.is_tensor(ii) else 0)

    def _seq(name, i):
        v = b.get(name)
        if v is None:
            return None
        if isinstance(v, dict) and "values" in v:
            off = v["offsets"].tolist()
            return v["values"][off[i]:off[i + 1]]
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == n_seq:
            return v[i]
        return None

    def _seq_pos(i):
        v = b.get("position_ids")
        if isinstance(v, dict) and "values" in v:
            vv, off = v["values"], v["offsets"].tolist()
            if vv.dim() == 2 and vv.shape[0] in (3, 4):
                return vv[:, off[i]:off[i + 1]]
        return None

    def _norm(p):  # -> (channels, L) cpu long, robust to (C,1,L)/(1,C,L)/(C,L)/(L,C)
        if p is None:
            return None
        p = p.detach().cpu().long()
        if p.dim() == 3:
            if p.shape[0] in (3, 4) and p.shape[1] == 1:      # (C, 1, L)
                p = p[:, 0]
            elif p.shape[1] in (3, 4) and p.shape[0] == 1:    # (1, C, L)
                p = p[0]
            else:
                p = p.reshape(p.shape[0], -1)
        if p.dim() == 2 and p.shape[0] not in (3, 4) and p.shape[1] in (3, 4):  # (L, C) -> (C, L)
            p = p.transpose(0, 1)
        return p

    def _add_text_axis_if_needed(p, L):
        p = _norm(p)
        if p is None or p.dim() != 2:
            return p
        if p.shape[0] == 4:
            return p
        if p.shape[0] == 3:
            textp = torch.arange(L, dtype=torch.long).unsqueeze(0)
            return torch.cat((textp, p), dim=0)
        return p

    # bind get_rope_index (verl normally binds the model's onto the processor)
    grf = getattr(processor, "get_rope_index", None)
    if grf is None:
        m = getattr(hf, "model", hf)
        grf = getattr(m, "get_rope_index", None) or getattr(hf, "get_rope_index", None)
        if grf is not None:
            processor.get_rope_index = grf
    _p(f"[PROVE] get_rope_index bound on processor: {grf is not None}")
    try:
        from verl.utils.model import compute_vlm_position_ids
    except Exception as e:  # noqa: BLE001
        _p(f"[PROVE] cannot import compute_vlm_position_ids ({e!r})")
        compute_vlm_position_ids = None

    image_token_id = getattr(processor, "image_token_id", None)
    merge = processor.image_processor.merge_size
    grid_all, pix_all = b.get("image_grid_thw"), b.get("pixel_values")
    img_cursor = patch_cursor = 0

    for i in range(min(n_seq, 2)):  # first 2 seqs is enough to prove
        ids_i = _seq("input_ids", i).long()
        L = ids_i.numel()
        ids2 = ids_i.unsqueeze(0).to(device)
        attn = torch.ones((1, L), dtype=torch.long, device=device)
        pd = _norm(_seq_pos(i))
        n_img_tok = int((ids_i == image_token_id).sum()) if image_token_id is not None else 0
        seq_grids, seq_pix, consumed = [], [], 0
        while grid_all is not None and img_cursor < grid_all.shape[0] and consumed < n_img_tok:
            g = grid_all[img_cursor]
            patches = int(g[0] * g[1] * g[2])
            seq_grids.append(g)
            if pix_all is not None:
                seq_pix.append(pix_all[patch_cursor:patch_cursor + patches])
            patch_cursor += patches
            consumed += patches // (merge * merge)
            img_cursor += 1
        grid_i = torch.stack(seq_grids).to(device) if seq_grids else None
        pix_i = torch.cat(seq_pix).to(device) if seq_pix else None
        _p(f"\n[PROVE] ===== seq{i} L={L} n_img_tok={n_img_tok} grids={len(seq_grids)} =====")
        if grid_i is None:
            _p("[PROVE] no image grid for this seq; skipping")
            continue
        grid_budget = int(((grid_i[:, 0] * grid_i[:, 1] * grid_i[:, 2]) // (merge * merge)).sum().item())
        _p(f"[PROVE] image_token_count={n_img_tok} grid_budget_tokens={grid_budget} "
           f"extra_image_tokens={max(0, n_img_tok - grid_budget)}")

        def _capture_hf_recompute_positions():
            """Run HF without explicit position_ids and capture the actual patched-HF position path."""
            import inspect

            rope_calls = []
            compute3d_calls = []
            lm_calls = []
            patched = []
            seen = set()

            def _summarize_mmtt(mmtt):
                if mmtt is None:
                    return "absent"
                m = mmtt.detach().cpu()
                vals = torch.unique(m)
                counts = {int(v): int((m == v).sum()) for v in vals}
                return f"shape={tuple(m.shape)} counts={counts}"

            def _summarize_pos(pos):
                if pos is None:
                    return "absent"
                if not torch.is_tensor(pos):
                    return type(pos).__name__
                p = pos.detach().cpu()
                msg = f"shape={tuple(p.shape)} dtype={p.dtype}"
                if p.numel():
                    flat = p.reshape(-1)
                    msg += f" min={int(flat.min())} max={int(flat.max())}"
                    if p.dim() >= 1:
                        last = p.reshape(-1, p.shape[-1])
                        starts = int((last[:, 0] == 0).sum())
                        msg += f" first_col_zero_rows={starts}/{last.shape[0]}"
                return msg

            def _patch_rope(obj_name, obj):
                key = (id(obj), "get_rope_index")
                if obj is None or key in seen or not hasattr(obj, "get_rope_index"):
                    return
                seen.add(key)
                orig = getattr(obj, "get_rope_index")
                try:
                    sig = inspect.signature(orig)
                except Exception:  # noqa: BLE001
                    sig = None

                def wrapped(*args, **kwargs):
                    bound_args = {}
                    if sig is not None:
                        try:
                            bound_args = dict(sig.bind_partial(*args, **kwargs).arguments)
                        except Exception:  # noqa: BLE001
                            bound_args = {}
                    merged = dict(bound_args)
                    merged.update(kwargs)
                    out = orig(*args, **kwargs)
                    pos = out[0] if isinstance(out, (tuple, list)) else out
                    rope_calls.append(
                        {
                            "where": obj_name,
                            "n_args": len(args),
                            "keys": sorted(merged.keys()),
                            "mmtt": _summarize_mmtt(merged.get("mm_token_type_ids")),
                            "pos_shape": tuple(pos.shape) if torch.is_tensor(pos) else type(pos).__name__,
                            "pos": _add_text_axis_if_needed(pos, L) if torch.is_tensor(pos) else None,
                        }
                    )
                    return out

                setattr(obj, "get_rope_index", wrapped)
                patched.append((obj, "get_rope_index", orig))

            def _patch_compute3d(obj_name, obj):
                key = (id(obj), "compute_3d_position_ids")
                if obj is None or key in seen or not hasattr(obj, "compute_3d_position_ids"):
                    return
                seen.add(key)
                orig = getattr(obj, "compute_3d_position_ids")
                try:
                    sig = inspect.signature(orig)
                except Exception:  # noqa: BLE001
                    sig = None

                def wrapped(*args, **kwargs):
                    bound_args = {}
                    if sig is not None:
                        try:
                            bound_args = dict(sig.bind_partial(*args, **kwargs).arguments)
                        except Exception:  # noqa: BLE001
                            bound_args = {}
                    merged = dict(bound_args)
                    merged.update(kwargs)
                    out = orig(*args, **kwargs)
                    compute3d_calls.append(
                        {
                            "where": obj_name,
                            "n_args": len(args),
                            "keys": sorted(merged.keys()),
                            "attention_mask": _summarize_pos(merged.get("attention_mask")),
                            "pos_shape": tuple(out.shape) if torch.is_tensor(out) else type(out).__name__,
                            "pos": _add_text_axis_if_needed(out, L) if torch.is_tensor(out) else None,
                        }
                    )
                    return out

                setattr(obj, "compute_3d_position_ids", wrapped)
                patched.append((obj, "compute_3d_position_ids", orig))

            def _patch_lm_forward(obj_name, obj):
                key = (id(obj), "forward")
                if obj is None or not hasattr(obj, "forward") or key in seen:
                    return
                seen.add(key)
                orig = getattr(obj, "forward")

                def wrapped(*args, **kwargs):
                    pos = kwargs.get("position_ids")
                    lm_calls.append(
                        {
                            "where": obj_name,
                            "keys": sorted(kwargs.keys()),
                            "position_ids": _summarize_pos(pos),
                            "position_ids_tensor": _add_text_axis_if_needed(pos, L) if torch.is_tensor(pos) else None,
                            "cache_position": _summarize_pos(kwargs.get("cache_position")),
                            "attention_mask": _summarize_pos(kwargs.get("attention_mask")),
                            "inputs_embeds": _summarize_pos(kwargs.get("inputs_embeds")),
                        }
                    )
                    return orig(*args, **kwargs)

                setattr(obj, "forward", wrapped)
                patched.append((obj, "forward", orig))

            _patch_rope("hf", hf)
            _patch_rope("hf.model", getattr(hf, "model", None))
            _patch_rope("processor", processor)
            _patch_compute3d("hf", hf)
            _patch_compute3d("hf.model", getattr(hf, "model", None))
            _patch_lm_forward("hf.language_model", getattr(hf, "language_model", None))
            _patch_lm_forward("hf.model.language_model", getattr(getattr(hf, "model", None), "language_model", None))
            try:
                with torch.no_grad():
                    kw = {"input_ids": ids2, "use_cache": False}
                    if pix_i is not None:
                        kw["pixel_values"] = pix_i.to(hf.dtype)
                        kw["image_grid_thw"] = grid_i
                    hf(**kw)
            except Exception as e:  # noqa: BLE001
                _p(f"[PROVE] HF forward recompute-position run failed: {e!r}")
            finally:
                for obj, attr, orig in patched:
                    setattr(obj, attr, orig)

            if not rope_calls:
                _p("[PROVE] HF forward recompute did not call any captured get_rope_index attr")
            for cidx, call in enumerate(rope_calls):
                _p(f"[PROVE] HF get_rope_index call{cidx}@{call['where']}: n_args={call['n_args']} "
                   f"keys={call['keys']} mm_token_type_ids={call['mmtt']} out_shape={call['pos_shape']}")
            if not compute3d_calls:
                _p("[PROVE] HF forward recompute did not call any captured compute_3d_position_ids attr")
            for cidx, call in enumerate(compute3d_calls):
                _p(f"[PROVE] HF compute_3d_position_ids call{cidx}@{call['where']}: n_args={call['n_args']} "
                   f"keys={call['keys']} attention_mask={call['attention_mask']} out_shape={call['pos_shape']}")
            if not lm_calls:
                _p("[PROVE] HF forward recompute did not reach a captured language_model.forward")
            for cidx, call in enumerate(lm_calls):
                _p(f"[PROVE] HF language_model.forward call{cidx}@{call['where']}: keys={call['keys']} "
                   f"position_ids={call['position_ids']} cache_position={call['cache_position']} "
                   f"attention_mask={call['attention_mask']} inputs_embeds={call['inputs_embeds']}")
            out = {}
            if rope_calls:
                out["hf-get_rope"] = rope_calls[-1]["pos"]
            if compute3d_calls:
                out["hf-compute3d"] = compute3d_calls[-1]["pos"]
            lm_pos = next((call["position_ids_tensor"] for call in reversed(lm_calls) if call["position_ids_tensor"] is not None), None)
            if lm_pos is not None:
                out["hf-lm-pos"] = lm_pos
            elif lm_calls:
                # If the text model receives no explicit position_ids, its own fallback is
                # cache_position expanded to the three MRoPE axes. Compare that effective
                # fallback explicitly so the mismatch is visible.
                ar = torch.arange(L, dtype=torch.long)
                out["hf-lm-fallback"] = ar.unsqueeze(0).expand(4, -1)
            return out

        # V1 WAY: get_rope_index WITHOUT mm_token_type_ids, then prepend text axis (verbatim v1 logic)
        pv = None
        try:
            vp, _ = grf(input_ids=ids2, attention_mask=attn, image_grid_thw=grid_i)
            vp = vp.transpose(0, 1)  # (3,1,L) -> (1,3,L) as v1 does
            textp = torch.ones((1, L), dtype=torch.long, device=vp.device)
            textp[0] = torch.arange(L, device=vp.device)
            textp = textp.unsqueeze(0)  # (1,1,L)
            pv = _norm(torch.cat((textp, vp), dim=1))  # (4, L)
        except Exception as e:  # noqa: BLE001
            _p(f"[PROVE] v1-way get_rope_index failed: {e!r}")

        # V1 MARK-ALL WAY: get_rope_index WITH mm_token_type_ids that marks ALL image_token positions
        # (verbatim v1 agent_loop.py:991), vs async which marks only grid-budget tokens.
        pm = None
        try:
            mmtt = torch.zeros_like(ids2)
            if image_token_id is not None:
                mmtt[0][ids2[0] == image_token_id] = 1
            vp2, _ = grf(input_ids=ids2, attention_mask=attn, image_grid_thw=grid_i, mm_token_type_ids=mmtt)
            vp2 = vp2.transpose(0, 1)
            textp2 = torch.ones((1, L), dtype=torch.long, device=vp2.device)
            textp2[0] = torch.arange(L, device=vp2.device)
            pm = _norm(torch.cat((textp2.unsqueeze(0), vp2), dim=1))
        except Exception as e:  # noqa: BLE001
            _p(f"[PROVE] v1-markall get_rope_index failed: {e!r}")

        # ASYNC WAY: compute_vlm_position_ids (builds mm_token_type_ids by grid budget)
        pa = None
        if compute_vlm_position_ids is not None:
            try:
                pa = _norm(compute_vlm_position_ids(processor, ids2.cpu(), attn.cpu(), {"image_grid_thw": grid_i.cpu()}))
            except Exception as e:  # noqa: BLE001
                _p(f"[PROVE] async-way failed: {e!r}")

        # HF FORWARD RECOMPUTE: this is the exact path exercised by --no-position-ids.
        ph = _capture_hf_recompute_positions()

        def _cmp(name, x):
            if x is None or pd is None:
                _p(f"[PROVE] seq{i} {name}: n/a (x={x is not None}, dump={pd is not None})")
                return
            k = min(x.shape[-1], pd.shape[-1])
            d = (x[:, :k] - pd[:, :k]).abs()
            per_ch = d.max(dim=1).values.tolist()
            nmis = int((d.sum(0) > 0).sum())
            _p(f"[PROVE] {name:15s} vs DUMP: identical={nmis == 0} max_abs_per_channel(text,t,h,w)={per_ch} "
               f"mismatch_positions={nmis}/{k}")
            return nmis

        def _show_first_mismatch(name, x, nmis):
            if x is None or pd is None or not nmis:
                return
            k = min(x.shape[-1], pd.shape[-1])
            d = (x[:, :k] - pd[:, :k]).abs().sum(0)
            first = int((d > 0).nonzero(as_tuple=True)[0][0])
            w0, w1 = max(0, first - 1), min(k, first + 5)
            _p(f"[PROVE] {name} first mismatch at pos {first}; window [{w0}:{w1}] "
               f"(channels = text,t,h,w):")
            _p(f"[PROVE]   DUMP  =\n{pd[:, w0:w1]}")
            _p(f"[PROVE]   {name} =\n{x[:, w0:w1]}")

        nmis_async = _cmp("async(budget)", pa)
        nmis_markall = _cmp("v1-markall", pm)
        nmis_v1 = _cmp("v1-no-mmtt", pv)
        hf_pos_items = ph if isinstance(ph, dict) else {"hf-recompute": ph}
        nmis_hf = {name: _cmp(name, pos) for name, pos in hf_pos_items.items()}
        _show_first_mismatch("async(budget)", pa, nmis_async)
        _show_first_mismatch("v1-markall", pm, nmis_markall)
        _show_first_mismatch("v1-no-mmtt", pv, nmis_v1)
        for name, pos in hf_pos_items.items():
            _show_first_mismatch(name, pos, nmis_hf.get(name))

        # feed each to HF and show |A - C|
        A = _seq("old_log_probs", i)
        rm = _seq("response_mask", i)
        if A is None:
            continue
        A = A.float().reshape(-1)
        rm = rm.reshape(-1).bool() if rm is not None else None
        rv = int(rm.sum()) if rm is not None else A.numel()
        Av = A[rm] if (rm is not None and rm.numel() == A.numel()) else A[:rv]

        def _hf_ac(tag, pos):
            if pos is None:
                return
            with torch.no_grad():
                kw = {"input_ids": ids2, "use_cache": False, "position_ids": pos.to(device).unsqueeze(1)}
                if pix_i is not None:
                    kw["pixel_values"] = pix_i.to(hf.dtype)
                    kw["image_grid_thw"] = grid_i
                lp = torch.log_softmax(hf(**kw).logits[0].float(), dim=-1)
            C = torch.tensor([float(lp[L - rv + j - 1, int(ids_i[L - rv + j])]) for j in range(rv)])
            k = min(C.numel(), Av.numel())
            d = (Av[:k].cpu() - C[:k]).abs()
            m = torch.isfinite(d)
            _p(f"[PROVE] |A-C| with {tag:10s} positions: mean={d[m].mean():.4f} max={d[m].max():.4f}")

        _hf_ac("DUMP", pd)
        _hf_ac("async(budget)", pa)
        _hf_ac("v1-markall", pm)
        _hf_ac("v1-no-mmtt", pv)
        for name, pos in hf_pos_items.items():
            _hf_ac(name, pos)
    _p("\n[PROVE] READ: async(grid-budget) / v1-markall / DUMP identify whether trainer-side MRoPE construction")
    _p("[PROVE] matches the saved training positions. hf-recompute is the --no-position-ids path; under verl's")
    _p("[PROVE] patched Qwen3-VL forward it may not call processor/model.get_rope_index at all, which means")
    _p("[PROVE] the mismatch is 'no explicit training MRoPE positions were fed', not necessarily a bad")
    _p("[PROVE] mm_token_type_ids calculation. If async/v1-markall == DUMP but --no-position-ids has a large")
    _p("[PROVE] logprob gap, treat that gap as a microbench recompute/fallback artifact.")


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

        # Replay the rollout's ACTUAL sampling params when the dump saved them (new dumps do; old ones fall
        # back to raw/temperature=1). logprobs_mode (raw vs processed) is an ENGINE arg -> pass to LLM().
        rsp = b.get("sampling_params") if isinstance(b.get("sampling_params"), dict) else {}
        # logprobs_mode is a top-level dump field (rollout ENGINE config), NOT in sampling_params. If the
        # dump lacks it (old dump), DON'T silently assume raw — the rollout default is PROCESSED, so a wrong
        # guess makes A-D differ. Warn loudly.
        logprobs_mode = b.get("logprobs_mode")
        if logprobs_mode is None:
            logprobs_mode = "raw_logprobs"
            _p("[ROLLOUT-BUNDLE] WARNING: logprobs_mode NOT in dump (old dump). Defaulting raw_logprobs, but the "
               "rollout run may have used 'processed_logprobs' -> A-D may not match. Re-dump to capture it.")
        temp = float(rsp.get("temperature", 1.0))
        _p(f"[ROLLOUT-BUNDLE] replay sampling: logprobs_mode={logprobs_mode} temperature={temp} "
           f"(sampling_params in dump: {bool(rsp)}; logprobs_mode in dump: {b.get('logprobs_mode') is not None})")
        _p("[ROLLOUT-BUNDLE] loading vLLM for fresh recompute (D)...")
        llm_kwargs = dict(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max(1, len(images))},
            enforce_eager=True,
        )
        try:
            llm = LLM(logprobs_mode=logprobs_mode, **llm_kwargs)
        except TypeError:  # older vLLM: logprobs_mode not an LLM arg
            _p("[ROLLOUT-BUNDLE] (this vLLM build doesn't take logprobs_mode= on LLM(); using default)")
            llm = LLM(**llm_kwargs)
        sp = SamplingParams(temperature=temp if temp > 0 else 1.0, max_tokens=1, prompt_logprobs=1)
        # byte-exact prompt replay if the dump saved vLLM's own expanded ids; else pass my `full` + image.
        _vpids_saved = b.get("vllm_prompt_token_ids")
        req = {"prompt_token_ids": [int(x) for x in _vpids_saved] if _vpids_saved else full}
        if images:
            req["multi_modal_data"] = {"image": images}
        out = llm.generate([req], sp)[0]
        pl = out.prompt_logprobs  # list aligned with vLLM's ACTUAL (image-expanded) prompt tokens
        # Align via vLLM's OWN expanded prompt_token_ids (the image placeholder expands, so my `full` and
        # vLLM's prompt differ in length; the response tokens are the LAST n of BOTH). Index by vpids so
        # position and token id are self-consistent -- no offset guessing.
        vpids = [int(x) for x in out.prompt_token_ids] if out.prompt_token_ids is not None else full

        def _lp(entry, tok):
            if not entry:
                return float("nan")
            for k in (tok, str(tok)):  # some vLLM versions key prompt_logprobs by str token id
                if k in entry:
                    e = entry[k]
                    return float(getattr(e, "logprob", e))  # Logprob obj or raw float
            return float("nan")

        resp_ids_n = [int(x) for x in response_ids[:n]]
        tail = vpids[-n:]
        tail_ok = tail == resp_ids_n
        _p(f"[ROLLOUT-BUNDLE] vLLM expanded prompt len={len(vpids)} (my full={len(full)}); "
           f"response tail == recorded response_ids: {tail_ok}")
        if not tail_ok:
            # find where the recorded response actually starts in vpids (robustness)
            for s in range(len(vpids) - n, -1, -1):
                if vpids[s:s + n] == resp_ids_n:
                    tail = vpids[s:s + n]
                    _p(f"[ROLLOUT-BUNDLE] found response at vpids[{s}:{s + n}] (not the exact tail)")
                    base = s
                    break
            else:
                base = len(vpids) - n
                _p("[ROLLOUT-BUNDLE] WARNING: recorded response not found verbatim in vLLM prompt; using tail")
        else:
            base = len(vpids) - n
        Dvals = [_lp(pl[base + j] if pl is not None else None, vpids[base + j]) for j in range(n)]
        D = torch.tensor(Dvals, dtype=torch.float64)
        _p(f"[ROLLOUT-BUNDLE] D valid (non-nan): {int((~torch.isnan(D)).sum())}/{n}")
    except Exception as e:  # noqa: BLE001
        import traceback

        _p(f"[ROLLOUT-BUNDLE] fresh-vLLM (D) failed ({e!r}); reporting A vs C only")
        traceback.print_exc()

    A = A[:n]

    def _cmp(name, x, y):
        d = (x - y).abs()
        m = torch.isfinite(d)  # drop nan/inf so one bad token doesn't poison mean/max
        if int(m.sum()) == 0:
            _p(f"  {name}: no finite tokens (all nan/inf)")
            return
        dd = d[m]
        _p(f"  {name}: mean={dd.mean():.4f} max={dd.max():.4f} frac>0.1={(dd > 0.1).float().mean():.3f} "
           f"n_valid={int(m.sum())}/{m.numel()}")

    _p("\n[ROLLOUT-BUNDLE] ===== per-response-token comparison =====")
    if D is not None:
        _cmp("|A-D| recorded-vLLM vs FRESH-vLLM  ", A, D)
    if C is not None:
        _cmp("|A-C| recorded-vLLM vs FRESH-HF/FSDP", A, C)
    if C is not None and D is not None:
        _cmp("|C-D| FRESH-HF vs FRESH-vLLM (clean gap)", C, D)
    _print_k3_matrix(
        "ROLLOUT-BUNDLE",
        [
            {
                "seq": 0,
                "n": n,
                "A": A.detach().cpu(),
                "B": None,
                "C": C.detach().cpu() if C is not None else None,
                "D": D.detach().cpu() if D is not None else None,
            }
        ],
    )
    _p("[ROLLOUT-BUNDLE] B=dump-FSDP is not in a raw rollout bundle. A/B/C/D in one table requires a dump "
       "that stores raw images plus fsdp_log_probs, or a matched rollout bundle and FSDP probe for the same sample.")
    # worst tokens on the clean gap (finite only, so nan tokens don't sort to the top as garbage)
    if C is not None and D is not None:
        cd = (C - D).abs()
        fin = torch.isfinite(cd).nonzero(as_tuple=True)[0]
        order = fin[torch.argsort(cd[fin], descending=True)][:12]
        _p("  worst |C-D| tokens (finite only) (pos | A  C  D | token):")
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
    ap.add_argument("--verl-patch", dest="verl_patch", action="store_true", default=True,
                    help="Apply verl's monkey_patch to HF so its forward == the TRAINING FSDP forward (default on).")
    ap.add_argument("--no-verl-patch", dest="verl_patch", action="store_false",
                    help="Use STOCK transformers HF (the old behavior that differs from verl FSDP by ~0.28).")
    ap.add_argument("--no-position-ids", dest="feed_pos", action="store_false", default=True,
                    help="ASYNC-BUNDLE: do NOT feed the dumped training position_ids to HF (let HF recompute "
                    "MRoPE itself). Use to isolate whether position_ids (not the patch) closes the |B-C| gap.")
    ap.add_argument(
        "--stock-hf-c",
        dest="stock_hf_c",
        action="store_true",
        default=False,
        help="ASYNC-BUNDLE convenience mode: make C a pure stock transformers HF forward and do NOT feed "
        "dumped position_ids. Equivalent to --no-verl-patch --no-position-ids; tests whether stock HF's "
        "own Qwen3-VL position computation matches dumped FSDP.",
    )
    ap.add_argument("--prove-positions", dest="prove_positions", default=None,
                    help="Path to the async .pt bundle. PROVE the root cause: recompute position_ids the ASYNC "
                    "way (with mm_token_type_ids) and the V1 way (without) and compare BOTH to the dumped "
                    "ground-truth positions; then feed each to HF and show |A-C|. Shows exactly where v1 diverges.")
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
    if args.stock_hf_c:
        args.verl_patch = False
        args.feed_pos = False

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

    # Apply verl's monkey_patch so the HF forward == the TRAINING FSDP forward (patched qwen3_vl_base_forward
    # + _get_input_embeds etc.). Without this, HF is STOCK transformers, which differs from verl's FSDP by
    # ~0.28 (the gap we chased). With it, C should reproduce B (async FSDP). --no-verl-patch to compare stock.
    if args.verl_patch:
        try:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(hf, use_remove_padding=True, use_fused_kernels=False)
            _p("=== [VERL-PATCH] applied verl monkey_patch to HF -> C should now match TRAINING FSDP (B) ===")
        except Exception as e:  # noqa: BLE001
            _p(f"=== [VERL-PATCH] FAILED ({e!r}); HF stays STOCK transformers (C != training FSDP) ===")
    else:
        _p("=== [VERL-PATCH] disabled (--no-verl-patch): HF is STOCK transformers ===")
        if args.stock_hf_c:
            _p("=== [STOCK-HF-C] C uses stock HF forward and receives NO dumped position_ids; "
               "stock HF should compute Qwen3-VL positions from input_ids + image_grid_thw ===")

    # ---- BUNDLE MODES: run BOTH in one shot when given, then exit (no generation needed) -------------
    #   --async-bundle   : A=async-vLLM(old) / B=async-FSDP(fsdp) / C=fresh-HF   (HF only)
    #   --rollout-bundle : A=recorded-vLLM   / C=fresh-HF        / D=fresh-vLLM  (loads vLLM)
    if args.prove_positions:
        _p("\n################## PROVE-POSITIONS (async-way vs v1-way vs dumped ground truth) ##################")
        try:
            _run_prove_positions(args.prove_positions, hf, processor, tokenizer, device)
        except Exception as e:  # noqa: BLE001
            import traceback

            _p(f"[PROVE] FAILED: {e!r}")
            traceback.print_exc()
        return

    if args.async_bundle or args.rollout_bundle:
        if args.async_bundle:
            _p("\n################## ASYNC-BUNDLE (A=async vLLM / B=async FSDP / C=fresh HF) ##################")
            try:
                _run_async_bundle(args.async_bundle, hf, processor, tokenizer, device, feed_pos=args.feed_pos)
            except Exception as e:  # noqa: BLE001 - don't let it block the rollout-bundle below
                import traceback

                _p(f"[ASYNC-BUNDLE] FAILED: {e!r}")
                traceback.print_exc()
        if args.rollout_bundle:
            _p("\n################## ROLLOUT-BUNDLE (A=recorded vLLM / C=fresh HF / D=fresh vLLM) ##################")
            try:
                _run_rollout_bundle(args.rollout_bundle, args.model, hf, processor, tokenizer, device, args.gpu_mem)
            except Exception as e:  # noqa: BLE001
                import traceback

                _p(f"[ROLLOUT-BUNDLE] FAILED: {e!r}")
                traceback.print_exc()
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
