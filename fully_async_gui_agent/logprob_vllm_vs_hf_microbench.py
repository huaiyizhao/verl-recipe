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
