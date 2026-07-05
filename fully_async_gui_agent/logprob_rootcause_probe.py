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
        default="sdpa",
        help="HF attn_implementation: sdpa/flash_attention_2 (match training's flash) or eager. "
        "If HF-vs-FSDP shrinks vs eager, the gap was an attention-kernel artifact, not a training bug.",
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

    _log(f"=== loading model {model_path} (attn={args.attn}) ===")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn, trust_remote_code=True
        ).eval()
    except Exception as e:  # noqa: BLE001
        _log(f"[warn] attn_implementation={args.attn} failed ({e!r}); falling back to eager")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
        ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_seq = min(args.max_seq, len(input_ids) if input_ids else 0)
    _log(f"=== analyzing {n_seq} sequences ===")

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
        pix_i = torch.cat(seq_pixels).to(device).to(model.dtype) if seq_pixels else None
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

        # HF forward (bf16, eager) with the model's OWN recomputed position_ids.
        def _hf_logp(dtype, ids, pix, grid):
            m = model if dtype == torch.bfloat16 else model.to(torch.float32)
            with torch.no_grad():
                kw = {"input_ids": ids.unsqueeze(0)}
                if pix is not None:
                    kw["pixel_values"] = pix.to(m.dtype)
                    kw["image_grid_thw"] = grid
                out = m(**kw, use_cache=False)
                logits = out.logits[0].float()  # (L, V)
                lp = torch.log_softmax(logits[:-1], dim=-1)
                per_tok = lp.gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)  # logp of actual next token, len L-1
            if dtype != torch.bfloat16:
                model.to(torch.bfloat16)
            return per_tok.cpu()

        try:
            hf_bf16 = _hf_logp(torch.bfloat16, ids_i, pix_i, grid_i)
            hf_fp32 = _hf_logp(torch.float32, ids_i, pix_i, grid_i)
        except Exception as e:  # noqa: BLE001
            _log(f"[FORWARD] HF forward failed: {e!r}; skipping numeric forks for this seq")
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

        def _mad(a, ref):
            if a is None or ref is None:
                return float("nan")
            n = min(a.numel(), ref.numel())
            return (a[:n].float() - ref[:n].float()).abs().mean().item()

        _log("[FORK] mean |Δ| over RESPONSE (assistant) tokens vs HF-fp32 (clean reference):")
        _log(f"   HF-bf16   vs HF-fp32 : {_mad(hf16_m, hf32_m):.4f}   (>0 => bf16 precision)")
        _log(f"   FSDP-train vs HF-fp32: {_mad(fsdp_m, hf32_m):.4f}   (HF unreliable at image tokens)")
        _log(f"   vLLM      vs HF-fp32 : {_mad(v_m, hf32_m):.4f}   (HF unreliable at image tokens)")

        # THE REAL QUESTION: where do vLLM(gen) and FSDP(recompute) — both REAL — diverge? That per-token
        # k3 is exactly what the RS mask thresholds on (seq-mean k3 > 0.005 -> whole sequence masked).
        if v_m is not None and fsdp_m is not None:
            n = min(v_m.numel(), fsdp_m.numel())
            vf = v_m[:n].float() - fsdp_m[:n].float()
            k3 = (torch.exp(vf.clamp(-20, 20)) - 1.0 - vf).abs()
            # bf16 self-swing: |HF-bf16 - HF-fp32| per token. Same (even if wrong) image, so this cleanly
            # measures how bf16-SENSITIVE each token's logp is, independent of reconstruction errors.
            # If the vLLM-FSDP gap tracks this bf16 swing -> the divergence is bf16 numerical (ROOT A).
            if hf16_m is not None and hf32_m is not None:
                nb = min(n, hf16_m.numel(), hf32_m.numel())
                bf16sw_all = (hf16_m[:nb].float() - hf32_m[:nb].float()).abs()
                max_bf16sw = float(bf16sw_all.max()) if nb else float("nan")
            else:
                bf16sw_all, max_bf16sw = None, float("nan")
            _log(
                f"[VF] vLLM-vs-FSDP over {n} resp tok: mean|Δ|={vf.abs().mean():.4f} max|Δ|={vf.abs().max():.4f} "
                f"seq_mean_k3={k3.mean():.5f} (mask@0.005 -> {'MASKED' if k3.mean() > 0.005 else 'kept'}) "
                f"n(|Δ|>0.1)={int((vf.abs() > 0.1).sum())} | pure-bf16 max_swing={max_bf16sw:.4f}"
            )
            order = torch.argsort(vf.abs(), descending=True)[: args.worst_k]
            ids_m = _sel(resp_ids.float(), mask).long() if resp_ids is not None else None
            _log("[WORST vLLM-FSDP] token |  vLLM    FSDP    Δ(v-f)   k3   bf16swing | decoded")
            for j in order.tolist():
                tid = int(ids_m[j].item()) if (ids_m is not None and j < ids_m.numel()) else -1
                txt = processor.tokenizer.decode([tid]) if tid >= 0 else "?"
                bsw = bf16sw_all[j].item() if (bf16sw_all is not None and j < bf16sw_all.numel()) else float("nan")
                _log(
                    f"   id={tid:6d} | {v_m[j].item():+.3f}  {fsdp_m[j].item():+.3f}  "
                    f"{vf[j].item():+.3f}  {k3[j].item():.3f}  {bsw:.3f} | {txt!r}"
                )

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
