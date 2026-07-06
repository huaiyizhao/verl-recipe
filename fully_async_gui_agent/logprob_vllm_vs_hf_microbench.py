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
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    _p(f"=== loading HF model (bf16, attn={args.attn}) — this IS the FSDP training-path numerics ===")
    hf = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn, trust_remote_code=True
    ).eval().to(device)

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
        # vLLM generate on the same raw image
        out = llm.generate([{"prompt": prompt_text, "multi_modal_data": {"image": img}}], sp)[0].outputs[0]
        gen_ids = list(out.token_ids)
        vllm_lp = [out.logprobs[i][gen_ids[i]].logprob for i in range(len(gen_ids))]
        full = torch.cat([prompt_ids, torch.tensor(gen_ids, dtype=prompt_ids.dtype)])
        hf_lp = hf_logp_for(full, start=prompt_ids.numel(), pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        _p(f"[IMAGE] response: {tokenizer.decode(gen_ids)[:300]!r}")
        results["IMAGE"] = _summ("IMAGE", vllm_lp, hf_lp, gen_ids, tokenizer)

    # ---------- VERDICT ----------
    _p("\n########## VERDICT ##########")
    tmean, tmax = results["TEXT"]
    _p(f"TEXT : mean_abs_d={tmean:.4f} max_abs_d={tmax:.4f}")
    if "IMAGE" in results:
        imean, imax = results["IMAGE"]
        _p(f"IMAGE: mean_abs_d={imean:.4f} max_abs_d={imax:.4f}")
        if tmean < 0.02 and imean > 3 * max(tmean, 1e-6):
            _p(">> VISION path: text agrees, image diverges -> the gap is driven by the image/vision forward.")
        elif tmean > 0.05:
            _p(">> GENERAL numerics: text ALSO diverges -> attention backend / mrope / logit-scale, NOT vision.")
        else:
            _p(">> Inconclusive: compare magnitudes above; also compare IMAGE(here) vs the pipeline LOGPROB_GAP.")
    else:
        _p("(no --image; run again with --image to get the vision-vs-text split.)")


if __name__ == "__main__":
    main()
