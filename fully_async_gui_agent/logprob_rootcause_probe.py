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
    ap.add_argument("--max-seq", type=int, default=4, help="how many sequences to analyze")
    ap.add_argument("--worst-k", type=int, default=8, help="worst-divergence tokens to print per sequence")
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
    input_ids = _unpack(b.get("input_ids"))  # list[1D] per seq
    resp_mask = _unpack(b.get("response_mask"))
    vllm_lp = _unpack(b.get("old_log_probs")) or _unpack(b.get("rollout_log_probs"))
    fsdp_lp = _unpack(b.get("fsdp_log_probs"))
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

    _log(f"=== loading model {model_path} (eager attn) ===")
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

        # (B1) MROPE CHECK: reference position_ids from CURRENT ids+grid vs the fed ones.
        if get_rope_index is not None and grid_i is not None:
            try:
                ref_pos = get_rope_index(processor, ids_i.cpu(), image_grid_thw=grid_i.cpu())  # (3 or 4, L)
                _log(f"[MROPE] ref_pos shape={tuple(ref_pos.shape)}  fed_pos raw shape printed above")
                # best-effort align: compare the fed positions for this seq if we can slice them.
                # (fed layout varies; print ref so a human can eyeball vs the dumped fed tensor)
                _log(f"[MROPE] ref_pos[:, :12]=\n{ref_pos[:, :12]}")
                _log(f"[MROPE] ref_pos[:, -12:]=\n{ref_pos[:, -12:]}")
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

        # align vLLM / FSDP-train logp (stored per response token) to HF next-token logp.
        rm = resp_mask[si] if resp_mask else None
        v = vllm_lp[si] if vllm_lp else None
        f = fsdp_lp[si] if fsdp_lp else None
        _log(f"[ALIGN] hf_next_len={hf_fp32.numel()} resp_mask_len={rm.numel() if rm is not None else None} "
             f"vllm_len={v.numel() if v is not None else None} fsdp_len={f.numel() if f is not None else None}")

        # response positions in the L-1 next-token frame: mask over tokens 1..L-1
        if rm is not None and rm.numel() == L:
            sel = rm[1:].bool()
        elif rm is not None and rm.numel() == L - 1:
            sel = rm.bool()
        else:
            sel = torch.ones(L - 1, dtype=torch.bool)
        hf32_r = hf_fp32[sel]
        hf16_r = hf_bf16[sel]

        def _mean_abs(a, ref):
            if a is None:
                return float("nan")
            n = min(a.numel(), ref.numel())
            return (a[:n].float() - ref[:n].float()).abs().mean().item()

        v_f = _mean_abs(v, f) if (v is not None and f is not None) else float("nan")
        _log("[FORK] mean |Δ| vs HF-fp32 (the clean reference):")
        _log(f"   HF-bf16   vs HF-fp32 : {_mean_abs(hf16_r, hf32_r):.4f}   (>0 => bf16 precision matters)")
        _log(f"   FSDP-train vs HF-fp32: {_mean_abs(f, hf32_r):.4f}   (>0 => train-path: remove_padding/FA/FSDP)")
        _log(f"   vLLM      vs HF-fp32 : {_mean_abs(v, hf32_r):.4f}   (>0 => vLLM engine differs from clean HF)")
        _log(f"   vLLM      vs FSDP    : {v_f:.4f}   (the gap that drives the mask)")

        # worst tokens by |vLLM - HF-fp32|
        if v is not None:
            n = min(v.numel(), hf32_r.numel())
            d = (v[:n].float() - hf32_r[:n].float()).abs()
            order = torch.argsort(d, descending=True)[: args.worst_k]
            resp_ids = ids_i[1:][sel][:n]
            _log("[WORST] token | vLLM  HF-fp32  HF-bf16  FSDP  |  decoded")
            for j in order.tolist():
                tid = int(resp_ids[j].item())
                txt = processor.tokenizer.decode([tid])
                fv = f[j].item() if (f is not None and j < f.numel()) else float("nan")
                _log(f"   id={tid:6d} | {v[j].item():+.3f}  {hf32_r[j].item():+.3f}  "
                     f"{hf16_r[j].item():+.3f}  {fv:+.3f}  | {txt!r}")

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
