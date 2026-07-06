#!/usr/bin/env python3
"""Offline root-cause analyzer for the vLLM<->FSDP logprob divergence.

Run this ON THE GPU BOX after ONE training run that dumped a probe bundle
(set VERL_LOGPROB_PROBE_DUMP=1; bundle lands in VERL_LOGPROB_PROBE_DIR, default
/tmp/logprob_probe/probe_rank0_0.pt). It reproduces the FSDP recompute in a clean
HF forward under a matrix of conditions and runs the decision tree that isolates
WHY the same token gets different logprobs from vLLM (generation) vs FSDP (recompute):

  (B1) mrope position_ids mismatch  -> structural, our recompute feeds wrong 3D positions
  (B2) training-path numerics        -> HF-fp32 == vLLM but FSDP-train != vLLM (bf16/remove_padding/FA/FSDP)
  (A)  bf16 precision                -> HF-fp32 != HF-bf16 and HF-bf16 ~ FSDP
  (C)  fundamental vLLM<->HF          -> HF-fp32 == FSDP-train but BOTH != vLLM (vLLM paged-attn/chunked-prefill/KV-dtype)

Usage:
  python logprob_rootcause_probe.py [/path/to/probe_rank0_0.pt] [--model /efs/data/models/Qwen3-VL-8B-Instruct]

It only READS the bundle and the checkpoint; it changes nothing. Heavy prints by design.
"""

import argparse
import os
import sys

import torch


def _log(msg):
    print(msg, flush=True)


def _unpack(field):
    """A dumped field is either a plain CPU tensor, a {values, offsets} nested dict, or None."""
    if field is None:
        return None
    if isinstance(field, dict) and "values" in field and "offsets" in field:
        vals, offs = field["values"], field["offsets"].tolist()
        return [vals[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]
    return field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="/tmp/logprob_probe/probe_rank0_0.pt")
    ap.add_argument("--model", default=None, help="HF model dir (defaults to model_path stored in the bundle)")
    ap.add_argument("--max-seq", type=int, default=8, help="how many sequences to analyze")
    ap.add_argument("--worst-k", type=int, default=8, help="worst-divergence tokens to print per sequence")
    ap.add_argument(
        "--attn",
        default="flash_attention_2",
        help="HF attn_implementation. Default flash_attention_2 to MATCH verl's flash_attn_varlen "
        "(fp32 accumulation). Falls back sdpa->eager if unavailable.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.bundle):
        _log(f"bundle not found: {args.bundle}")
        sys.exit(1)
    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    _log("=== bundle keys / shapes ===")
    for k, v in b.items():
        if isinstance(v, dict) and "values" in v:
            _log(f"  {k}: nested values={tuple(v['values'].shape)} offsets_n={v['offsets'].numel()}")
        elif torch.is_tensor(v):
            _log(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            _log(f"  {k}: {v!r}")

    model_path = args.model or b.get("model_path") or "/efs/data/models/Qwen3-VL-8B-Instruct"
    input_ids = _unpack(b.get("input_ids"))  # list[1D] per seq (FULL sequence: prompt + response)
    responses = _unpack(b.get("responses"))  # per seq: response token ids (response-length)
    resp_mask = _unpack(b.get("response_mask"))  # per seq: 1=assistant token, 0=tool token (response-length)
    vllm_lp = _unpack(b.get("old_log_probs")) or _unpack(b.get("rollout_log_probs"))  # response-length
    fsdp_lp = _unpack(b.get("fsdp_log_probs"))  # FULL-length per-token logp (rolled: pos p -> logp of token p+1)
    pos_fed_raw = b.get("position_ids")  # {values:(channels,total), offsets}; sliced per seq below
    grid_all = b.get("image_grid_thw")
    pixels_all = b.get("pixel_values")
    if isinstance(grid_all, dict):
        grid_all = grid_all["values"]
    if isinstance(pixels_all, dict):
        pixels_all = pixels_all["values"]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image_token_id = getattr(processor, "image_token_id", None)
    merge = processor.image_processor.merge_size

    try:
        from verl.models.transformers.qwen3_vl import get_rope_index
    except Exception as e:  # noqa: BLE001
        _log(f"[warn] could not import verl get_rope_index ({e!r}); mrope check will be skipped")
        get_rope_index = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(attn):
        try:
            m = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation=attn, trust_remote_code=True
            ).eval()
            return m.to(device)
        except Exception as e:  # noqa: BLE001
            _log(f"[warn] load attn={attn} failed ({e!r})")
            return None

    # sdpa model supports fp32/fp16/bf16 (flash-attn cannot do fp32). One model, cast per-precision.
    _log(f"=== loading model {model_path} (sdpa) ===")
    model_sdpa = _load("sdpa") or _load("eager")
    if model_sdpa is None:
        _log("could not load sdpa/eager model")
        sys.exit(1)
    # flash model (bf16) to run HF UNPACKED with the SAME kernel FSDP uses -> isolates packing from
    # kernel. If unpacked-flash == unpacked-sdpa but packed-FSDP differs, the bug is the PACKING.
    _log("=== loading flash_attention_2 model (bf16, unpacked ref) ===")
    model_flash = _load("flash_attention_2")
    _log(f"=== flash model available: {model_flash is not None} ===")

    n_seq = min(args.max_seq, len(input_ids) if input_ids else 0)
    _log(f"=== analyzing {n_seq} sequences ===")

    # Accumulators for the final PRECISION MATRIX, over RESPONSE (assistant) tokens of the GUI image
    # sequences. All aligned to the same tokens so we can diff each engine/precision vs HF-fp32 "truth".
    ACC = {"hf32": [], "hf16": [], "hffp16": [], "fsdp": [], "vllm": []}

    # split concatenated pixel_values across images by patch count (grid_t*grid_h*grid_w per image);
    # images appear in sequence order, so we walk grids and consume patches + assign to sequences.
    def _images_for_seq(ids_i):
        if image_token_id is None:
            return 0
        return int((ids_i == image_token_id).sum().item()) * (merge * merge)

    img_cursor = 0  # index into grid_all
    patch_cursor = 0

    def _pos_fed(si):
        # fed position_ids sliced for seq si: (channels, seq_len). offsets index the last (token) dim.
        if not isinstance(pos_fed_raw, dict):
            return None
        vals, offs = pos_fed_raw["values"], pos_fed_raw["offsets"].tolist()
        return vals[:, offs[si] : offs[si + 1]]

    for si in range(n_seq):
        ids_i = input_ids[si].to(device)
        L = ids_i.numel()
        _log(f"\n----- seq {si}: len={L} -----")

        # how many images this sequence holds (by placeholder-token count vs grid patch counts)
        n_img_tokens = int((ids_i == image_token_id).sum().item()) if image_token_id is not None else 0
        seq_grids, seq_pixels = [], []
        consumed_tokens = 0
        while grid_all is not None and img_cursor < grid_all.shape[0] and consumed_tokens < n_img_tokens:
            g = grid_all[img_cursor]
            patches = int(g[0] * g[1] * g[2])
            seq_grids.append(g)
            if pixels_all is not None:
                seq_pixels.append(pixels_all[patch_cursor : patch_cursor + patches])
            patch_cursor += patches
            consumed_tokens += patches // (merge * merge)
            img_cursor += 1
        grid_i = torch.stack(seq_grids).to(device) if seq_grids else None
        pix_i = torch.cat(seq_pixels).to(device) if seq_pixels else None  # _hf_logp re-casts per model
        _log(
            f"[IMG] n_img_tokens={n_img_tokens} images_assigned={len(seq_grids)} "
            f"patches={pix_i.shape[0] if pix_i is not None else 0} "
            f"(expect patches/merge^2 == n_img_tokens: {(pix_i.shape[0] // (merge * merge)) if pix_i is not None else 0})"
        )

        # (B1) MROPE CHECK: reference position_ids from CURRENT ids+grid vs the FED ones (diff, not eyeball).
        if get_rope_index is not None and grid_i is not None:
            try:
                ref_pos = get_rope_index(processor, ids_i.cpu(), image_grid_thw=grid_i.cpu())  # (3, L)
                fed = _pos_fed(si)  # (channels, L) or None
                if fed is not None:
                    # fed is 4-channel, ref is 3-channel. Qwen3-VL mrope order can be [t,h,w,*] or
                    # [*,t,h,w]; try both alignments and report the BEST (min) diff so a channel-order
                    # difference isn't mistaken for a position bug. Also print slices to eyeball.
                    fedc = fed.long().cpu()
                    refc = ref_pos.long().cpu()
                    cand = {"fed[:3]": fedc[:3], "fed[1:4]": fedc[1:4]}
                    best_name, best_mx, best_nmis = None, None, None
                    for name, fsub in cand.items():
                        if fsub.shape != refc.shape:
                            continue
                        d = (refc - fsub).abs()
                        mx = int(d.max())
                        if best_mx is None or mx < best_mx:
                            best_name, best_mx, best_nmis = name, mx, int((d > 0).sum())
                    _log(
                        f"[MROPE] ref{tuple(ref_pos.shape)} vs fed{tuple(fed.shape)}: "
                        f"best_align={best_name} max_abs_diff={best_mx} n_mismatch={best_nmis}"
                    )
                    _log(f"[MROPE] fed[:, :6]=\n{fedc[:, :6]}")
                    _log(f"[MROPE] fed[:, -6:]=\n{fedc[:, -6:]}")
                else:
                    _log(f"[MROPE] fed pos unavailable; ref shape={tuple(ref_pos.shape)}")
            except Exception as e:  # noqa: BLE001
                _log(f"[MROPE] failed: {e!r}")

        # HF forward -> per-token next-token logp (len L-1). cast_dtype temporarily casts the model
        # (fp32/fp16/bf16). Same forward each time, only compute precision differs. None on failure.
        def _hf_logp(mdl, ids, pix, grid, cast_dtype=None):
            if mdl is None:
                return None
            try:
                if cast_dtype is not None:
                    mdl.to(cast_dtype)
                with torch.no_grad():
                    kw = {"input_ids": ids.unsqueeze(0)}
                    if pix is not None:
                        kw["pixel_values"] = pix.to(mdl.dtype)
                        kw["image_grid_thw"] = grid
                    out = mdl(**kw, use_cache=False)
                    logits = out.logits[0].float()  # (L, V)
                    lp = torch.log_softmax(logits[:-1], dim=-1)
                    per = lp.gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)  # logp of actual next token
                return per.cpu()
            except Exception as e:  # noqa: BLE001
                _log(f"[FORWARD] failed ({e!r})")
                return None
            finally:
                if cast_dtype is not None:
                    mdl.to(torch.bfloat16)

        # Same model+input+attention; ONLY compute precision differs. fp32 = the "truth".
        hf_bf16 = _hf_logp(model_sdpa, ids_i, pix_i, grid_i)  # bf16 sdpa, UNPACKED
        hf_fp32 = _hf_logp(model_sdpa, ids_i, pix_i, grid_i, cast_dtype=torch.float32)  # fp32 truth
        hf_fp16 = _hf_logp(model_sdpa, ids_i, pix_i, grid_i, cast_dtype=torch.float16)  # fp16 (proposed)
        hf_flash = _hf_logp(model_flash, ids_i, pix_i, grid_i)  # bf16 FLASH, UNPACKED (same kernel as FSDP)
        if hf_bf16 is None or hf_fp32 is None or hf_fp16 is None:
            _log("[FORWARD] sdpa forward failed; skipping seq")
            continue

        # ALIGN: response = the LAST resp_len tokens (contiguous tail = everything after the initial
        # prompt). vLLM logp & responses are response-length; FSDP logp & HF logp are full-length
        # (next-token frame). Response token j == input_ids[prompt_len+j], its next-token logp sits at
        # frame position prompt_len-1+j. So the response window in the L-1 frame is [prompt_len-1 : L-1].
        v = vllm_lp[si] if vllm_lp else None
        rm = resp_mask[si] if resp_mask else None
        resp_ids = responses[si] if responses else None
        resp_len = (
            v.numel() if v is not None else (resp_ids.numel() if resp_ids is not None else 0)
        )
        prompt_len = L - resp_len
        lo, hi = prompt_len - 1, L - 1

        if resp_ids is not None:
            tail = ids_i[prompt_len:].cpu()
            ok = bool(tail.numel() == resp_ids.numel() and torch.equal(tail, resp_ids.cpu()))
            _log(f"[ALIGN] resp_len={resp_len} prompt_len={prompt_len} tail==responses:{ok}")
            if not ok:
                _log("[ALIGN] WARNING: response is not a contiguous tail (multi-turn?) — numbers below suspect")

        hf32_r = hf_fp32[lo:hi]
        hf16_r = hf_bf16[lo:hi]
        f_full = fsdp_lp[si] if fsdp_lp else None
        fsdp_r = f_full[lo:hi] if (f_full is not None and f_full.numel() >= hi) else f_full

        # DIAGNOSTIC (answers: is HF-vs-FSDP broken on ANY token, or only image/response?).
        # fsdp_log_probs is full-length, so compare HF vs FSDP over the WHOLE sequence and split by
        # region. If prompt generic-TEXT tokens also diverge -> systematic forward/alignment bug (any
        # seq). If only image-adjacent/response diverge -> multimodal-data specific.
        if f_full is not None and f_full.numel() >= L - 1:
            n2 = L - 1
            d_all = (hf_fp32[:n2].float() - f_full[:n2].float()).abs()  # fp32-HF vs bf16-FSDP
            d_bf = (hf_bf16[:n2].float() - f_full[:n2].float()).abs()  # bf16-HF vs bf16-FSDP
            # PRECISION error vs the fp32 "truth" (same model+input+attention, ONLY compute dtype changes).
            # THE fp16 test: if fp16's error vs truth << bf16's, fp16 preserves the distribution -> if we
            # move BOTH train+infer to fp16, they'd both stay near truth => match each other => mismatch fixed.
            err_bf = (hf_bf16[:n2].float() - hf_fp32[:n2].float()).abs()  # bf16 error vs fp32-truth
            err_16 = (hf_fp16[:n2].float() - hf_fp32[:n2].float()).abs()  # fp16 error vs fp32-truth
            nxt = ids_i[1 : n2 + 1].cpu()  # token predicted at each position p (= ids[p+1])
            is_img = (nxt == image_token_id) if image_token_id is not None else torch.zeros(n2, dtype=torch.bool)
            posn = torch.arange(n2)
            is_resp = posn >= (prompt_len - 1)
            imgpos = is_img.nonzero().flatten()
            first_img = int(imgpos[0]) if imgpos.numel() else n2
            is_pre = (posn < first_img) & (~is_img)  # pure text, no image seen yet
            is_post = (~is_img) & (~is_resp) & (posn >= first_img)
            unc_all = (hf_fp32[:n2].abs() > 1.0) & (~is_img)  # high-entropy tokens (sampled prob < e^-1)

            def _rm(msk, dv=d_all):
                return dv[msk].mean().item() if bool(msk.any()) else float("nan")

            _log(
                f"[REGION] mean|Δ| FSDP-vs-HF(fp32/bf16): "
                f"pre_text={_rm(is_pre, d_all):.4f}/{_rm(is_pre, d_bf):.4f}(n={int(is_pre.sum())}) "
                f"post_text={_rm(is_post, d_all):.4f}/{_rm(is_post, d_bf):.4f}(n={int(is_post.sum())}) "
                f"response={_rm(is_resp & ~is_img, d_all):.4f}/{_rm(is_resp & ~is_img, d_bf):.4f}"
            )
            # === THE FP16 TEST: does fp16 track the fp32 truth far better than bf16? ===
            _log(
                f"[FP16-TEST seq{si}] |HF_x - HF_fp32truth| (same forward, only precision differs): "
                f"ALL_txt bf16={_rm(~is_img, err_bf):.4f} fp16={_rm(~is_img, err_16):.4f} | "
                f"UNCERTAIN(n={int(unc_all.sum())}) bf16={_rm(unc_all, err_bf):.4f} fp16={_rm(unc_all, err_16):.4f} | "
                f"MAX_token bf16={err_bf[~is_img].max().item():.3f} fp16={err_16[~is_img].max().item():.3f}  "
                f"[fp16<<bf16 => fp16 preserves precision]"
            )

            # Comprehensive characterization of the PURE-TEXT (pre-image) FSDP-vs-HFfp32 gap: signed
            # stats (constant offset?), OLS FSDP~a*HF+b (scaling?), split by token confidence, and raw
            # token dumps (first-in-order + worst). Only seq0 to avoid spam.
            if si == 0 and bool(is_pre.any()):
                pi = is_pre.nonzero().flatten()
                fs = f_full[:n2][pi].float()
                hf = hf_fp32[:n2][pi].float()
                sd = fs - hf  # signed FSDP - HF
                # OLS FSDP ~ a*HF + b
                hm, fm = hf.mean(), fs.mean()
                a = ((hf - hm) * (fs - fm)).sum() / ((hf - hm) ** 2).sum().clamp(min=1e-9)
                b = fm - a * hm
                resid = fs - (a * hf + b)
                r2 = 1 - (resid.var() / fs.var().clamp(min=1e-9))
                conf = hf.abs() < 0.1  # near-certain tokens
                unc = hf.abs() > 1.0  # uncertain tokens
                _log(
                    f"[PRETEXT-STATS] n={pi.numel()} signed_mean={sd.mean().item():+.4f} std={sd.std().item():.4f} "
                    f"median={sd.median().item():+.4f} |max|={sd.abs().max().item():.3f} | "
                    f"OLS FSDP~{a.item():.4f}*HF+{b.item():+.4f} R2={r2.item():.4f} | "
                    f"|Δ|@confident={d_all[:n2][pi][conf].mean().item() if bool(conf.any()) else float('nan'):.4f} "
                    f"|Δ|@uncertain={d_all[:n2][pi][unc].mean().item() if bool(unc.any()) else float('nan'):.4f}"
                )
                # worst-12 pre-text tokens by FSDP-vs-fp32truth: show fp32/bf16/fp16 + per-token bf16-err
                # vs fp16-err, so you can SEE fp16 tracking the truth token-by-token where bf16 doesn't.
                worst = pi[torch.argsort(d_all[:n2][pi], descending=True)[:12]]
                _log("[PRETEXT worst-12] pos | FSDP    fp32    bf16    fp16   | bf16err fp16err | pred")
                for p in worst.tolist():
                    _log(
                        f"   {p:4d} | {f_full[p].item():+.3f} {hf_fp32[p].item():+.3f} {hf_bf16[p].item():+.3f} "
                        f"{hf_fp16[p].item():+.3f} | {(hf_bf16[p] - hf_fp32[p]).abs().item():.3f}   "
                        f"{(hf_fp16[p] - hf_fp32[p]).abs().item():.3f}   | "
                        f"{processor.tokenizer.decode([int(nxt[p])])!r}"
                    )

        # keep only assistant tokens (mask==1); tool tokens within the response region are not trained.
        if rm is not None and rm.numel() == resp_len:
            mask = rm.bool()
        else:
            mask = torch.ones(resp_len, dtype=torch.bool)

        def _sel(x, msk):
            if x is None:
                return None
            n = min(x.numel(), msk.numel())
            return x[:n][msk[:n]]

        hf32_m, hf16_m, fsdp_m, v_m = _sel(hf32_r, mask), _sel(hf16_r, mask), _sel(fsdp_r, mask), _sel(v, mask)
        hf16fp16_m = _sel(hf_fp16[lo:hi], mask)  # fp16 logp on response tokens
        hfflash_m = _sel(hf_flash[lo:hi], mask) if hf_flash is not None else None  # bf16-flash UNPACKED
        # accumulate aligned response tokens for the final precision matrix (all same length/tokens)
        if all(x is not None for x in (hf32_m, hf16_m, hf16fp16_m, fsdp_m, v_m)):
            mlen = min(hf32_m.numel(), hf16_m.numel(), hf16fp16_m.numel(), fsdp_m.numel(), v_m.numel())
            ACC["hf32"].append(hf32_m[:mlen])
            ACC["hf16"].append(hf16_m[:mlen])
            ACC["hffp16"].append(hf16fp16_m[:mlen])
            ACC["fsdp"].append(fsdp_m[:mlen])
            ACC["vllm"].append(v_m[:mlen])

        def _mad(a, ref):
            if a is None or ref is None:
                return float("nan")
            n = min(a.numel(), ref.numel())
            return (a[:n].float() - ref[:n].float()).abs().mean().item()

        _log("[FORK] mean |Δ| over RESPONSE (assistant) tokens vs HF-fp32 (clean reference):")
        _log(f"   HF-bf16   vs HF-fp32 : {_mad(hf16_m, hf32_m):.4f}   (>0 => bf16 precision)")
        _log(f"   FSDP-train vs HF-fp32: {_mad(fsdp_m, hf32_m):.4f}   (HF unreliable at image tokens)")
        _log(f"   vLLM      vs HF-fp32 : {_mad(v_m, hf32_m):.4f}   (HF unreliable at image tokens)")

        # THE REAL QUESTION: where do vLLM(gen) and FSDP(recompute) diverge? That per-token k3 is what
        # the RS mask thresholds on (seq-mean k3 > 0.005 -> whole sequence masked). AND: is vLLM or FSDP
        # the outlier? FSDP and HF-bf16 are BOTH `transformers` (packed-flash vs unpacked-sdpa); vLLM is
        # its OWN impl. All bf16. If |FSDP-HFbf16| << |vLLM-HFbf16| -> the two transformers forwards agree
        # and vLLM is the outlier (impl/version). If the reverse -> FSDP's packed path is the outlier.
        if v_m is not None and fsdp_m is not None:
            n = min(v_m.numel(), fsdp_m.numel())
            vf = v_m[:n].float() - fsdp_m[:n].float()
            k3 = (torch.exp(vf.clamp(-20, 20)) - 1.0 - vf).abs()
            hi = vf.abs() > 0.1  # the mask-driving (diverging) tokens

            def _m2(a, b, msk=None):
                if a is None or b is None:
                    return float("nan")
                m = min(a.numel(), b.numel())
                d = (a[:m].float() - b[:m].float()).abs()
                if msk is not None:
                    d = d[msk[:m]]
                return d.mean().item() if d.numel() else float("nan")

            _log(
                f"[VF] vLLM-vs-FSDP over {n} resp tok: mean|Δ|={vf.abs().mean():.4f} max|Δ|={vf.abs().max():.4f} "
                f"seq_mean_k3={k3.mean():.5f} (mask@0.005 -> {'MASKED' if k3.mean() > 0.005 else 'kept'}) "
                f"n(|Δ|>0.1)={int(hi.sum())}"
            )
            _log(
                f"[IMPL-TEST] both bf16, vs HF-bf16(=transformers): "
                f"ALL FSDP-vs-HFbf16={_m2(fsdp_m, hf16_m):.4f} vLLM-vs-HFbf16={_m2(v_m, hf16_m):.4f} | "
                f"on |Δ|>0.1: FSDP-vs-HFbf16={_m2(fsdp_m, hf16_m, hi):.4f} vLLM-vs-HFbf16={_m2(v_m, hf16_m, hi):.4f} "
                f"[the BIGGER one is the outlier vs transformers]"
            )
            # PACKING vs KERNEL: HF-flash is UNPACKED + same flash kernel as FSDP. If HFflash≈HFsdpa
            # (kernel agrees unpacked) but FSDP(packed-flash) differs -> the bug is the PACKING/rmpad.
            if hfflash_m is not None:
                _log(
                    f"[PACK-TEST] on |Δ|>0.1 tokens: HFflash-vs-HFsdpa={_m2(hfflash_m, hf16_m, hi):.4f} "
                    f"(≈0 => flash kernel fine unpacked) | FSDP-vs-HFflash={_m2(fsdp_m, hfflash_m, hi):.4f} "
                    f"(large => PACKING is the bug, not the kernel)"
                )
            order = torch.argsort(vf.abs(), descending=True)[: args.worst_k]
            ids_m = _sel(resp_ids.float(), mask).long() if resp_ids is not None else None
            _log("[WORST vLLM-FSDP] token |  vLLM     FSDP    HFbf16   HFfp32 | Δ(v-f)   k3   | decoded")
            for j in order.tolist():
                tid = int(ids_m[j].item()) if (ids_m is not None and j < ids_m.numel()) else -1
                txt = processor.tokenizer.decode([tid]) if tid >= 0 else "?"
                hb = hf16_m[j].item() if (hf16_m is not None and j < hf16_m.numel()) else float("nan")
                hf = hf32_m[j].item() if (hf32_m is not None and j < hf32_m.numel()) else float("nan")
                _log(
                    f"   id={tid:6d} | {v_m[j].item():+.3f}  {fsdp_m[j].item():+.3f}  {hb:+.3f}  {hf:+.3f} "
                    f"| {vf[j].item():+.3f}  {k3[j].item():.3f} | {txt!r}"
                )

    # ---- SYNTHETIC PURE-TEXT case: run HF fp32/fp16/bf16 on a plain paragraph (no images) ----
    text = (
        "Reinforcement learning fine-tunes a policy by sampling trajectories, scoring them with a reward, "
        "and increasing the probability of high-reward actions. In practice the rollout engine and the "
        "training engine can disagree on token log-probabilities because of floating point rounding, which "
        "turns nominally on-policy updates into biased off-policy ones. The larger the mantissa, the smaller "
        "this disagreement becomes, so the choice of numeric format matters a great deal for stability."
    )
    txt_stats = None
    try:
        tids = processor.tokenizer(text, return_tensors="pt").input_ids[0].to(device)
        t32 = _hf_logp(model_sdpa, tids, None, None, cast_dtype=torch.float32)
        t16 = _hf_logp(model_sdpa, tids, None, None, cast_dtype=torch.float16)
        tbf = _hf_logp(model_sdpa, tids, None, None)
        if t32 is not None and t16 is not None and tbf is not None:
            uncm = t32.abs() > 1.0  # high-entropy text tokens
            txt_stats = {
                "bf16_all": (tbf - t32).abs().mean().item(),
                "fp16_all": (t16 - t32).abs().mean().item(),
                "bf16_unc": (tbf - t32).abs()[uncm].mean().item() if bool(uncm.any()) else float("nan"),
                "fp16_unc": (t16 - t32).abs()[uncm].mean().item() if bool(uncm.any()) else float("nan"),
                "bf16_max": (tbf - t32).abs().max().item(),
                "fp16_max": (t16 - t32).abs().max().item(),
                "n": int(tids.numel()),
                "n_unc": int(uncm.sum()),
            }
    except Exception as e:  # noqa: BLE001
        _log(f"[TEXT] synthetic-text forward failed ({e!r})")

    # ---- FINAL PRECISION MATRIX: deviation from HF-fp32 "truth" ----
    _log("\n" + "=" * 78)
    _log("=== PRECISION MATRIX: mean |logp - HF_fp32_truth| (lower = closer to truth) ===")
    _log("    Engine/precision      | image-data(resp)           | text-data(synthetic)")
    _log("                          | ALL      UNCERTAIN   MAXtok | ALL      UNCERTAIN   MAXtok")

    def _fmt(a, ref, mask=None):
        if a is None or ref is None:
            return "  n/a  "
        d = (a.float() - ref.float()).abs()
        if mask is not None:
            d = d[mask]
        return f"{d.mean().item():.4f}" if d.numel() else "  n/a  "

    if ACC["hf32"]:
        H32 = torch.cat(ACC["hf32"])
        H16 = torch.cat(ACC["hf16"])
        HF16 = torch.cat(ACC["hffp16"])
        FS = torch.cat(ACC["fsdp"])
        VL = torch.cat(ACC["vllm"])
        um = H32.abs() > 1.0  # high-entropy image-response tokens
        img_n, img_unc = H32.numel(), int(um.sum())

        def row(name, arr, tstats_key=None):
            iall = _fmt(arr, H32)
            iunc = _fmt(arr, H32, um)
            imax = f"{(arr.float() - H32.float()).abs().max().item():.3f}" if arr is not None else " n/a "
            if tstats_key and txt_stats:
                tall = f"{txt_stats[tstats_key + '_all']:.4f}"
                tunc = f"{txt_stats[tstats_key + '_unc']:.4f}"
                tmax = f"{txt_stats[tstats_key + '_max']:.3f}"
            else:
                tall = tunc = tmax = " n/a "
            _log(f"    {name:21s} | {iall}   {iunc}    {imax}  | {tall}   {tunc}    {tmax}")

        _log(f"    (image n={img_n}, uncertain={img_unc}; text n={txt_stats['n'] if txt_stats else '?'}, "
             f"uncertain={txt_stats['n_unc'] if txt_stats else '?'})")
        row("HF-fp32 (truth)", H32, None)  # 0 by definition
        row("HF-fp16", HF16, "fp16")
        row("HF-bf16", H16, "bf16")
        row("FSDP-bf16 (bundle)", FS, None)
        row("vLLM-bf16 (bundle)", VL, None)
        _log("    ------------------------------------------------------------------")
        _log("    NOTE: FSDP/vLLM only exist at bf16 here (that's what training ran). Their fp16/fp32")
        _log("    rows need a real engine rerun; the HF fp16-vs-bf16 gap PREDICTS them. fp32 is")
        _log("    infeasible in flash-attn/vLLM -> it's only the reference 'truth', not a runnable config.")
    else:
        _log("    (no accumulated tokens)")
    _log("=" * 78)

    _log(
        "\n=== VERDICT GUIDE ===\n"
        " MROPE ref != fed         -> ROOT (B1): mrope position_ids (structural). Fix the recompute's positions.\n"
        " FSDP-train != HF-fp32,\n"
        "   vLLM ~ HF-fp32         -> ROOT (B2): training-path numerics (remove_padding/FA/FSDP), NOT fundamental.\n"
        " HF-bf16 != HF-fp32,\n"
        "   FSDP ~ HF-bf16         -> ROOT (A): bf16 precision amplified at flat softmax. Mitigate w/ fp32 logp or TIS.\n"
        " HF-fp32 ~ FSDP-train,\n"
        "   BOTH != vLLM           -> ROOT (C): fundamental vLLM<->HF (paged-attn/chunked-prefill/KV dtype).\n"
        "                             No FSDP change fixes it; use vLLM-prefill as reference or TIS."
    )


if __name__ == "__main__":
    main()
