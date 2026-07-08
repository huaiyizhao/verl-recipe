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
import math

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


def _load_hf_model(model_path, attn, device):
    from transformers import AutoModelForImageTextToText

    _p(f"=== loading HF model (bf16, attn={attn}) — this IS the FSDP training-path numerics ===")
    hf = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn, trust_remote_code=True
    ).eval().to(device)

    try:
        from verl.models.transformers.monkey_patch import apply_monkey_patch

        apply_monkey_patch(hf, use_remove_padding=True, use_fused_kernels=False)
        _p("=== [VERL-PATCH] applied verl monkey_patch to HF -> C matches TRAINING FSDP path ===")
    except Exception as e:  # noqa: BLE001
        _p(f"=== [VERL-PATCH] FAILED ({e!r}); C may not match training FSDP ===")
    return hf


def _collapse_image_runs(ids, image_token_id):
    """vLLM wants one image placeholder per image; dumped input_ids have one per image patch token."""
    out, n_runs, prev = [], 0, False
    for t in ids:
        t = int(t)
        is_img = image_token_id is not None and t == image_token_id
        if is_img:
            if not prev:
                out.append(t)
                n_runs += 1
        else:
            out.append(t)
        prev = is_img
    return out, n_runs


def _vllm_seq_logprobs(llm, sampling_params, unexpanded_full, images, response_ids, n):
    """Teacher-force one sequence through vLLM and return response-token prompt logprobs."""
    req = {"prompt_token_ids": [int(x) for x in unexpanded_full]}
    if images:
        req["multi_modal_data"] = {"image": images}
    out = llm.generate([req], sampling_params)[0]
    prompt_logprobs = out.prompt_logprobs
    vpids = [int(x) for x in out.prompt_token_ids] if out.prompt_token_ids is not None else list(unexpanded_full)

    def _lp(entry, tok):
        if not entry:
            return float("nan")
        for k in (tok, str(tok)):
            if k in entry:
                e = entry[k]
                return float(getattr(e, "logprob", e))
        return float("nan")

    resp_ids_n = [int(x) for x in response_ids[:n]]
    base = len(vpids) - n
    if n and vpids[-n:] != resp_ids_n:
        for s in range(len(vpids) - n, -1, -1):
            if vpids[s:s + n] == resp_ids_n:
                base = s
                break
        else:
            _p("[FRESH-vLLM] WARNING: response not found verbatim in vLLM prompt; using tail")
    return torch.tensor(
        [_lp(prompt_logprobs[base + j] if prompt_logprobs is not None else None, vpids[base + j]) for j in range(n)],
        dtype=torch.float64,
    )


def _processor_param(processor, name, default=None):
    for obj in (getattr(processor, "image_processor", None), getattr(processor, "video_processor", None)):
        if obj is not None and hasattr(obj, name):
            v = getattr(obj, name)
            if v is not None:
                return v
    return default


def _as_3(v, default):
    if v is None:
        v = default
    if isinstance(v, (int, float)):
        return [float(v), float(v), float(v)]
    return [float(x) for x in v]


def _pixels_to_reconstructed_images(pixel_chunks, grid_chunks, processor):
    """Invert Qwen2/Qwen3-VL patchify+normalize into resized PIL images for fresh vLLM replay.

    This reconstructs the post-resize uint8 image represented by pixel_values. It cannot recover the
    original screenshot dimensions/crop, but reprocessing this image should reproduce the same visual
    patches up to dtype/rounding noise.
    """
    from PIL import Image

    patch_size = int(_processor_param(processor, "patch_size", 0) or 0)
    temporal_patch_size = int(_processor_param(processor, "temporal_patch_size", 2) or 2)
    merge_size = int(_processor_param(processor, "merge_size", None) or _processor_param(processor, "spatial_merge_size", 2) or 2)
    do_normalize = bool(_processor_param(processor, "do_normalize", True))
    do_rescale = bool(_processor_param(processor, "do_rescale", True))
    rescale_factor = float(_processor_param(processor, "rescale_factor", 1 / 255.0) or (1 / 255.0))
    mean = _as_3(_processor_param(processor, "image_mean", None), [0.5, 0.5, 0.5])
    std = _as_3(_processor_param(processor, "image_std", None), [0.5, 0.5, 0.5])

    images = []
    for pv, grid in zip(pixel_chunks, grid_chunks, strict=False):
        pv = pv.detach().cpu().float()
        t, h, w = [int(x) for x in grid.detach().cpu().reshape(-1)[:3].tolist()]
        dim = int(pv.shape[-1])
        if patch_size <= 0 or 3 * temporal_patch_size * patch_size * patch_size != dim:
            if patch_size > 0 and dim % (3 * patch_size * patch_size) == 0:
                temporal_patch_size = dim // (3 * patch_size * patch_size)
            else:
                patch_size = int(round(math.sqrt(dim / (3 * temporal_patch_size))))
        expected_dim = 3 * temporal_patch_size * patch_size * patch_size
        if expected_dim != dim:
            raise ValueError(
                f"cannot invert pixel_values dim={dim}; inferred patch={patch_size}, temporal={temporal_patch_size}"
            )
        n_patches = t * h * w
        if pv.shape[0] < n_patches:
            raise ValueError(f"pixel chunk too short: got {pv.shape[0]} patches, need {n_patches}")
        ghm, gwm = h // merge_size, w // merge_size
        if ghm * merge_size != h or gwm * merge_size != w:
            raise ValueError(f"grid h/w must be divisible by merge_size: grid={(t, h, w)} merge={merge_size}")

        x = pv[:n_patches].reshape(
            t, ghm, gwm, merge_size, merge_size, 3, temporal_patch_size, patch_size, patch_size
        )
        # Forward patchify order is (t, gh/m, gw/m, m, m, c, tp, p, p). Invert back to frames, C, H, W.
        frames = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()
        frames = frames.reshape(t * temporal_patch_size, 3, h * patch_size, w * patch_size)
        if do_normalize:
            mean_t = torch.tensor(mean, dtype=frames.dtype).view(1, 3, 1, 1)
            std_t = torch.tensor(std, dtype=frames.dtype).view(1, 3, 1, 1)
            frames = frames * std_t + mean_t
        if do_rescale:
            frames = frames / rescale_factor
        arr = frames[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def _rs_k3(rollout_logp, training_logp):
    """K3 in the same direction as verl rollout RS: d = logp_training - logp_rollout."""
    n = min(rollout_logp.numel(), training_logp.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.float64)
    rollout = rollout_logp[:n].to(torch.float64)
    training = training_logp[:n].to(torch.float64)
    d = (training - rollout).clamp(-20, 20)
    out = torch.exp(d) - 1.0 - d
    return out[torch.isfinite(out)]


def _rs_k3_mean(rollout_logp, training_logp):
    v = _rs_k3(rollout_logp, training_logp)
    return float(v.mean()) if v.numel() else float("nan")


def _fmt(v):
    return "nan" if not torch.isfinite(torch.tensor(v)) else f"{v:.5f}"


def _abs_stats(x, y):
    if x is None or y is None:
        return None
    n = min(x.numel(), y.numel())
    if n == 0:
        return None
    d = (x[:n].to(torch.float64) - y[:n].to(torch.float64)).abs()
    m = torch.isfinite(d)
    if int(m.sum()) == 0:
        return None
    dd = d[m]
    return float(dd.mean()), float(dd.max()), int(m.sum()), int(m.numel())


def _fmt_abs(x, y):
    s = _abs_stats(x, y)
    return "mean=nan max=nan n=0/0" if s is None else f"mean={s[0]:.4f} max={s[1]:.4f} n={s[2]}/{s[3]}"


def _gather_logp_from_logits(logits, labels, temperature=1.0):
    if temperature is None or temperature <= 0:
        temperature = 1.0
    x = logits / float(temperature) if abs(float(temperature) - 1.0) > 1e-8 else logits
    return torch.log_softmax(x, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _target_logit_lse_from_logits(logits, labels, temperature=1.0):
    """Return selected target logits and logsumexp on the same scaled-logit frame as logprob."""
    if temperature is None or temperature <= 0:
        temperature = 1.0
    x = logits.float()
    if abs(float(temperature) - 1.0) > 1e-8:
        x = x / float(temperature)
    labels = labels.long()
    target_logits = x.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.logsumexp(x, dim=-1)
    return target_logits, logsumexp


def _print_k3_matrix(tag, seq_rows):
    """Print seq-level pairwise K3 in verl rollout-RS direction."""
    pairs = [
        ("A", "B"),
        ("A", "Bprobe"),
        ("A", "Bref"),
        ("A", "BrefProbe"),
        ("A", "C"),
        ("A", "Cpack"),
        ("A", "D"),
        ("B", "Bprobe"),
        ("B", "Bref"),
        ("B", "BrefProbe"),
        ("B", "C"),
        ("B", "Cpack"),
        ("B", "D"),
        ("Bprobe", "BrefProbe"),
        ("Bprobe", "C"),
        ("Bprobe", "Cpack"),
        ("Bref", "BrefProbe"),
        ("Bref", "C"),
        ("Bref", "Cpack"),
        ("BrefProbe", "C"),
        ("BrefProbe", "Cpack"),
        ("C", "Cpack"),
        ("C", "D"),
        ("Cpack", "D"),
    ]
    have_any = False
    _p(f"\n[{tag}] ===== seq-level RS-K3 approximate KL =====")
    _p(f"[{tag}] convention: K3(rollout||training)=mean(exp(logp_training-logp_rollout)-1-(logp_training-logp_rollout))")
    for row in seq_rows:
        vals = []
        for a, b in pairs:
            if row.get(a) is not None and row.get(b) is not None:
                vals.append(f"K3({a}||{b})={_fmt(_rs_k3_mean(row[a], row[b]))}")
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
            agg[(a, b)] = _rs_k3_mean(torch.cat(xs), torch.cat(ys))
    if agg:
        _p(f"[{tag}] aggregate " + " ".join(f"K3({a}||{b})={_fmt(v)}" for (a, b), v in agg.items()))
    if not have_any:
        _p(f"[{tag}] no pairwise streams available for k3")


def _run_async_bundle(path, model_path, hf, processor, tokenizer, device, feed_pos=True, with_vllm=True,
                      attn="flash_attention_2",
                      gpu_mem=0.6, max_seq=0):
    """Load an async-run probe bundle and 3-way compare, per response token:
        A = async's recorded old_log_probs  (vLLM, since bypass_mode sets old = rollout_log_probs)
        B = async's recorded fsdp_log_probs (the async FSDP forward during that run)
        C = a FRESH HF/FSDP recompute here, on the SAME tokens+pixels.
        D = optional FRESH vLLM recompute; if raw_images are absent, reconstruct resized images from pixel_values.
    |A-B| reproduces async's tiny recorded gap. |A-C| is the KEY: if small, async's "vLLM" old_log_prob
    actually matches a fresh FSDP forward on real GUI data (=> the real vLLM<->FSDP gap is small, and the
    microbench's 0.28 was a verbose-prompt artifact); if ~0.28, async old IS vLLM-with-vision-gap.
    |B-C| is a sanity check (async FSDP vs fresh HF; should be ~bf16 noise)."""
    _p(f"=== [ASYNC-BUNDLE] loading {path} ===")
    b = torch.load(path, map_location="cpu", weights_only=False)
    if hf is None:
        hf = _load_hf_model(model_path, attn, device)

    # Fields can be nested ({values, offsets}) OR plain padded tensors (n_seq, ...). Print + handle both.
    for k in ("input_ids", "position_ids", "responses", "old_log_probs", "rollout_log_probs", "fsdp_log_probs",
              "fsdp_log_probs_ref", "response_mask", "pixel_values", "image_grid_thw", "logprobs_mode",
              "probe_maxk3", "trainer_global_steps", "probe_parameter_sync_step", "probe_batch_tags", "fsdp_probe",
              "fsdp_param_fingerprint"):
        v = b.get(k)
        if isinstance(v, dict) and "values" in v:
            _p(f"  {k}: nested values={tuple(v['values'].shape)} n_off={len(v['offsets'])}")
        elif torch.is_tensor(v):
            _p(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            _p(f"  {k}: dict keys={sorted(v.keys())}")
        elif isinstance(v, list):
            _p(f"  {k}: list len={len(v)}")
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
    if isinstance(pix, dict) and "values" in pix:
        pix = pix["values"]
    if isinstance(grid, dict) and "values" in grid:
        grid = grid["values"]
    raw_images = b.get("raw_images")
    fsdp_probe = b.get("fsdp_probe") if isinstance(b.get("fsdp_probe"), dict) else None
    logprobs_mode = b.get("logprobs_mode") or "raw_logprobs"
    temperature = b.get("temperature")
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(hf, "config", None), "image_token_id", None)
    merge = int(_processor_param(processor, "merge_size", None) or _processor_param(processor, "spatial_merge_size", 2) or 2)
    _p(f"[ASYNC-BUNDLE] n_seq={n_seq}")
    if b.get("trainer_global_steps") is not None or b.get("probe_batch_tags") is not None:
        tags = b.get("probe_batch_tags") or []
        gs = b.get("trainer_global_steps")
        pss = b.get("probe_parameter_sync_step")
        tag_steps = [
            (t.get("global_steps"), t.get("min_global_steps"), t.get("max_global_steps"))
            for t in tags
            if isinstance(t, dict)
        ]
        _p(f"[ASYNC-BUNDLE] trainer_global_steps={gs} parameter_sync_step={pss} "
           f"tag_steps(first8 global/min/max)={tag_steps[:8]}")
    if fsdp_probe is not None:
        path = fsdp_probe.get("path")
        reason = fsdp_probe.get("reason")
        n_probe = int(fsdp_probe["rows"].numel()) if torch.is_tensor(fsdp_probe.get("rows")) else 0
        n_valid_probe = int(fsdp_probe["valid_mask"].bool().sum()) if torch.is_tensor(fsdp_probe.get("valid_mask")) else 0
        _p(f"[ASYNC-BUNDLE] fsdp_probe path={path} reason={reason} response_rows={n_probe} valid={n_valid_probe}")
    fp = b.get("fsdp_param_fingerprint")
    if isinstance(fp, list) and fp:
        fp_desc = [
            f"{x.get('name')}:{x.get('dtype')}:{tuple(x.get('local_shape', ()))}/"
            f"sum={float(x.get('local_sum', float('nan'))):+.3e}"
            for x in fp[:5]
            if isinstance(x, dict)
        ]
        _p(f"[ASYNC-BUNDLE] fsdp_param_fingerprint(first5 rank0-local): {fp_desc}")

    def _probe_seq(name, i, valid_only=True):
        if fsdp_probe is None:
            return None
        seq_ids = fsdp_probe.get("seq_ids")
        x = fsdp_probe.get(name)
        if not torch.is_tensor(seq_ids) or not torch.is_tensor(x):
            return None
        m = seq_ids.long() == int(i)
        valid = fsdp_probe.get("valid_mask")
        if valid_only and torch.is_tensor(valid):
            m = m & valid.bool()
        if x.shape[0] != m.numel():
            return None
        return x[m].detach().cpu()

    img_cursor, patch_cursor = 0, 0
    accA, accB, accC, accD = [], [], [], []
    seq_rows = []
    d_jobs = []
    pack_jobs = []

    def _temp_for_seq(i):
        if temperature is None:
            return 1.0
        try:
            if torch.is_tensor(temperature):
                if temperature.dim() == 0:
                    return float(temperature.item())
                if temperature.shape[0] == n_seq:
                    return float(temperature[i].reshape(-1)[0].item())
                return float(temperature.reshape(-1)[0].item())
            if isinstance(temperature, (list, tuple)):
                return float(temperature[i] if i < len(temperature) else temperature[0])
            return float(temperature)
        except Exception:  # noqa: BLE001
            return 1.0

    def _run_packed_hf_recompute():
        """Replay the same dumped sequences as one remove-padding packed HF forward.

        C is the clean per-sequence HF replay. Cpack/Cp is the missing control for the
        trainer path: concatenate the same rows into a single (1, total_len) sequence,
        feed the dumped MRoPE position_ids as (channels, 1, total_len), and let the
        verl Qwen3-VL monkey patch inject varlen cu_seq_lens exactly like training.
        """
        if not pack_jobs:
            return
        bad_pos = [j["row"]["seq"] for j in pack_jobs if j.get("pos") is None]
        if bad_pos:
            _p(f"[ASYNC-BUNDLE] skipping fresh-HF packed(Cp): missing dumped position_ids for seqs={bad_pos}")
            return
        ids_parts, pos_parts, pix_parts, grid_parts = [], [], [], []
        for job in pack_jobs:
            ids = job["ids"].long().reshape(-1)
            pos = job["pos"].long()
            if pos.dim() != 2 or pos.shape[-1] != ids.numel() or pos.shape[0] not in (3, 4):
                _p(f"[ASYNC-BUNDLE] skipping fresh-HF packed(Cp): seq{job['row']['seq']} "
                   f"position_ids shape={tuple(pos.shape)} ids_len={ids.numel()}")
                return
            ids_parts.append(ids)
            pos_parts.append(pos)
            pix_parts.extend(job["seq_pix"])
            grid_parts.extend(job["seq_grids"])
        ids_cat = torch.cat(ids_parts).to(device)
        pos_cat = torch.cat(pos_parts, dim=-1).to(device)
        _p(f"[ASYNC-BUNDLE] running fresh-HF packed(Cp): n_seq={len(pack_jobs)} "
           f"total_len={ids_cat.numel()} position_ids={tuple(pos_cat.shape)}")
        try:
            with torch.no_grad():
                kw = {
                    "input_ids": ids_cat.unsqueeze(0),
                    "attention_mask": None,
                    "use_cache": False,
                    "position_ids": pos_cat.unsqueeze(1),  # (channels, batch=1, total_len)
                }
                if pix_parts:
                    kw["pixel_values"] = torch.cat(pix_parts).to(device).to(hf.dtype)
                    kw["image_grid_thw"] = torch.stack(grid_parts).to(device)
                logits_pack = hf(**kw).logits[0].float()
        except Exception as e:  # noqa: BLE001
            import traceback

            _p(f"[ASYNC-BUNDLE] fresh-HF packed(Cp) failed: {e!r}")
            traceback.print_exc()
            return

        offset = 0
        for job in pack_jobs:
            row = job["row"]
            ids = job["ids"].long().reshape(-1)
            L = ids.numel()
            if job["align_mode"] == "response-window":
                prompt_len = int(job["prompt_len"])
                resp_len = int(job["resp_len"])
                lo, hi = offset + prompt_len - 1, offset + L - 1
                target = ids[prompt_len:L].to(device)
                mask = job["mask"]
                if not torch.is_tensor(mask):
                    mask = torch.ones(resp_len, dtype=torch.bool)
                mask = mask.bool().reshape(-1)
                Cp0 = _gather_logp_from_logits(logits_pack[lo:hi], target, temperature=1.0).detach().cpu()
                Cpt0 = _gather_logp_from_logits(
                    logits_pack[lo:hi], target, temperature=float(job["temperature"])
                ).detach().cpu()
                CpTgt0, CpLse0 = _target_logit_lse_from_logits(logits_pack[lo:hi], target, temperature=1.0)
                CptTgt0, CptLse0 = _target_logit_lse_from_logits(
                    logits_pack[lo:hi], target, temperature=float(job["temperature"])
                )
                CpTgt0, CpLse0 = CpTgt0.detach().cpu(), CpLse0.detach().cpu()
                CptTgt0, CptLse0 = CptTgt0.detach().cpu(), CptLse0.detach().cpu()
                n = min(Cp0.numel(), mask.numel())
                Cp = Cp0[:n][mask[:n]]
                Cpt = Cpt0[:n][mask[:n]]
                CpTgt = CpTgt0[:n][mask[:n]]
                CpLse = CpLse0[:n][mask[:n]]
                CptTgt = CptTgt0[:n][mask[:n]]
                CptLse = CptLse0[:n][mask[:n]]
            else:
                rv = int(job["resp_valid"])
                lo, hi = offset + L - rv - 1, offset + L - 1
                target = ids[L - rv:L].to(device)
                Cp = _gather_logp_from_logits(logits_pack[lo:hi], target, temperature=1.0).detach().cpu()
                Cpt = _gather_logp_from_logits(
                    logits_pack[lo:hi], target, temperature=float(job["temperature"])
                ).detach().cpu()
                CpTgt, CpLse = _target_logit_lse_from_logits(logits_pack[lo:hi], target, temperature=1.0)
                CptTgt, CptLse = _target_logit_lse_from_logits(
                    logits_pack[lo:hi], target, temperature=float(job["temperature"])
                )
                CpTgt, CpLse = CpTgt.detach().cpu(), CpLse.detach().cpu()
                CptTgt, CptLse = CptTgt.detach().cpu(), CptLse.detach().cpu()
                mask = job["mask"]
                if torch.is_tensor(mask) and mask.numel() == Cp.numel():
                    Cp = Cp[mask.bool().reshape(-1)]
                    Cpt = Cpt[mask.bool().reshape(-1)]
                    CpTgt = CpTgt[mask.bool().reshape(-1)]
                    CpLse = CpLse[mask.bool().reshape(-1)]
                    CptTgt = CptTgt[mask.bool().reshape(-1)]
                    CptLse = CptLse[mask.bool().reshape(-1)]

            k = min(row["n"], Cp.numel(), Cpt.numel(), CpTgt.numel(), CpLse.numel(), CptTgt.numel(), CptLse.numel())
            if k > 0:
                for name in (
                    "A", "B", "Bprobe", "Bref", "BrefProbe", "C", "Ctemp", "tokens",
                    "FtargetIds", "FtargetLogit", "Flogsumexp", "CtargetLogit", "Clogsumexp",
                    "CtempTargetLogit", "CtempLogsumexp",
                ):
                    if row.get(name) is not None:
                        row[name] = row[name][:k]
                row["Cpack"] = Cp[:k]
                row["CpackTemp"] = Cpt[:k]
                row["CpackTargetLogit"] = CpTgt[:k]
                row["CpackLogsumexp"] = CpLse[:k]
                row["CpackTempTargetLogit"] = CptTgt[:k]
                row["CpackTempLogsumexp"] = CptLse[:k]
                row["n"] = k
            offset += L

    n_loop = min(n_seq, max_seq) if max_seq and max_seq > 0 else n_seq
    for i in range(n_loop):
        ids_i = _seq("input_ids", i)
        if ids_i is None:
            continue
        ids_i = ids_i.to(device).long().reshape(-1)
        L = ids_i.numel()
        A = _seq("old_log_probs", i)
        if A is None:
            A = _seq("rollout_log_probs", i)
        Bfull = _seq("fsdp_log_probs", i)
        BrefFull = _seq("fsdp_log_probs_ref", i)
        R = _seq("responses", i)
        rm = _seq("response_mask", i)
        A = A.float().reshape(-1) if A is not None else None
        Bfull = Bfull.float().reshape(-1) if Bfull is not None else None
        BrefFull = BrefFull.float().reshape(-1) if BrefFull is not None else None
        R = R.long().reshape(-1) if R is not None else None
        rm = rm.reshape(-1).bool() if rm is not None else None

        # Training computes rollout-correction on response-frame tensors:
        #   A / response_mask / responses: (response_len,)
        #   B / C: full next-token frame, where response token j at input_ids[prompt_len+j]
        #          is scored by frame prompt_len-1+j.
        # Therefore we must slice the full frame with response_len first, then apply response_mask.
        # Using response_mask.sum() as the response length shifts the window when masked/tool/pad tokens exist.
        resp_valid = int(rm.sum()) if rm is not None else (A.numel() if A is not None else 0)
        resp_len = None
        resp_len_src = "unknown"
        if R is not None and 0 < R.numel() < L:
            resp_len = int(R.numel())
            resp_len_src = "responses"
        elif A is not None and rm is not None and A.numel() == rm.numel() and int(rm.sum()) == A.numel() and 0 < A.numel() < L:
            # No masking/padding inside the response frame, so A/rm length is unambiguous.
            resp_len = int(A.numel())
            resp_len_src = "old_log_probs"
        elif A is not None and rm is None and 0 < A.numel() < L:
            resp_len = int(A.numel())
            resp_len_src = "old_log_probs"

        align_mode = "response-window"
        if resp_len is None:
            # Older async dumps did not save responses and may have padded old_log_probs to max_response_len
            # while input_ids are trimmed to actual tokens. In that ambiguous case, preserve the old tail-valid
            # behavior instead of inventing a prompt length.
            resp_len = resp_valid
            resp_len_src = "mask.sum fallback"
            align_mode = "valid-tail"

        if resp_len <= 0 or resp_len >= L:
            continue
        prompt_len = L - resp_len
        lo, hi = prompt_len - 1, L - 1
        if prompt_len < 1:
            continue

        if R is not None and R.numel() == resp_len:
            tail = ids_i[prompt_len:].detach().cpu()
            tail_ok = bool(tail.numel() == R.numel() and torch.equal(tail, R.cpu()))
            if not tail_ok:
                _p(f"[ASYNC-BUNDLE] seq{i} WARNING: input_ids response tail != dumped responses; "
                   f"alignment may be suspect (resp_len={resp_len}, prompt_len={prompt_len})")
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
        # fresh HF forward -> logp of the response-frame tokens
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
            logits = hf(**kw).logits[0].float()
        Dv = None
        Ctm = None
        if align_mode == "response-window":
            mask = rm[:resp_len].detach().cpu() if (rm is not None and rm.numel() >= resp_len) else torch.ones(resp_len, dtype=torch.bool)

            def _resp_frame(x):
                if x is None:
                    return None
                if x.numel() >= hi:
                    return x[lo:hi].detach().cpu()
                if x.numel() >= resp_len:
                    return x[:resp_len].detach().cpu()
                return None

            Av0 = A[:resp_len].detach().cpu() if (A is not None and A.numel() >= resp_len) else None
            Bv0 = _resp_frame(Bfull)
            Bref0 = _resp_frame(BrefFull)
            target = ids_i[prompt_len:L]
            frames = torch.arange(lo, hi, device=device)
            resp_logits = logits[frames]
            Cm0 = _gather_logp_from_logits(resp_logits, target, temperature=1.0).detach().cpu()
            Ct0 = _gather_logp_from_logits(resp_logits, target, temperature=_temp_for_seq(i)).detach().cpu()
            Ctgt0, Clse0 = _target_logit_lse_from_logits(resp_logits, target, temperature=1.0)
            Cttgt0, Ctlse0 = _target_logit_lse_from_logits(resp_logits, target, temperature=_temp_for_seq(i))
            Ctgt0, Clse0 = Ctgt0.detach().cpu(), Clse0.detach().cpu()
            Cttgt0, Ctlse0 = Cttgt0.detach().cpu(), Ctlse0.detach().cpu()

            def _masked(x):
                if x is None:
                    return None
                n = min(x.numel(), mask.numel())
                return x[:n][mask[:n]]

            Av = _masked(Av0)
            Bv = _masked(Bv0)
            Brefv = _masked(Bref0)
            Cm = _masked(Cm0)
            Ctm = _masked(Ct0)
            Ctgt = _masked(Ctgt0)
            Clse = _masked(Clse0)
            Cttgt = _masked(Cttgt0)
            Ctlse = _masked(Ctlse0)
            Tv = _masked(target.detach().cpu()).long()
        else:
            # Ambiguous older bundle: no full response frame is available, so compare valid tail tokens.
            mask = rm.detach().cpu() if rm is not None else None
            Av = (A[rm] if (A is not None and rm is not None and rm.numel() == A.numel())
                  else (A[:resp_valid] if A is not None else None))
            if Bfull is not None:
                if Bfull.numel() >= L - 1:
                    Bv = Bfull[L - resp_valid - 1:L - 1].detach().cpu()
                elif rm is not None and rm.numel() == Bfull.numel():
                    Bv = Bfull[rm].detach().cpu()
                else:
                    Bv = Bfull[-resp_valid:].detach().cpu()
            else:
                Bv = None
            if BrefFull is not None:
                if BrefFull.numel() >= L - 1:
                    Brefv = BrefFull[L - resp_valid - 1:L - 1].detach().cpu()
                elif rm is not None and rm.numel() == BrefFull.numel():
                    Brefv = BrefFull[rm].detach().cpu()
                else:
                    Brefv = BrefFull[-resp_valid:].detach().cpu()
            else:
                Brefv = None
            frames = torch.arange(L - resp_valid - 1, L - 1, device=device)
            target = ids_i[L - resp_valid:L]
            resp_logits = logits[frames]
            Cm = _gather_logp_from_logits(resp_logits, target, temperature=1.0).detach().cpu()
            Ctm = _gather_logp_from_logits(resp_logits, target, temperature=_temp_for_seq(i)).detach().cpu()
            Ctgt, Clse = _target_logit_lse_from_logits(resp_logits, target, temperature=1.0)
            Cttgt, Ctlse = _target_logit_lse_from_logits(resp_logits, target, temperature=_temp_for_seq(i))
            Ctgt, Clse = Ctgt.detach().cpu(), Clse.detach().cpu()
            Cttgt, Ctlse = Cttgt.detach().cpu(), Ctlse.detach().cpu()
            Tv0 = ids_i[L - resp_valid:L].detach().cpu()
            if torch.is_tensor(mask) and mask.numel() == Tv0.numel():
                m = mask.bool()
                Cm, Ctm = Cm[m], Ctm[m]
                Ctgt, Clse = Ctgt[m], Clse[m]
                Cttgt, Ctlse = Cttgt[m], Ctlse[m]
                Tv = Tv0[m]
            else:
                Tv = Tv0[:resp_valid]
        # defensive: align all to the common valid length
        k = min(
            x.numel()
            for x in (Cm,)
            + ((Av,) if Av is not None else ())
            + ((Bv,) if Bv is not None else ())
            + ((Brefv,) if Brefv is not None else ())
            + ((Ctm,) if Ctm is not None else ())
            + ((Ctgt,) if Ctgt is not None else ())
            + ((Clse,) if Clse is not None else ())
            + ((Cttgt,) if Cttgt is not None else ())
            + ((Ctlse,) if Ctlse is not None else ())
        )
        if k == 0:
            continue
        Cm = Cm[:k]
        Ctm = Ctm[:k] if Ctm is not None else None
        Am = Av[:k] if Av is not None else None
        Bm = Bv[:k] if Bv is not None else None
        Brefm = Brefv[:k] if Brefv is not None else None
        Tm = Tv[:k] if Tv is not None else None
        CtargetLogit = Ctgt[:k] if Ctgt is not None else None
        Clogsumexp = Clse[:k] if Clse is not None else None
        CtempTargetLogit = Cttgt[:k] if Cttgt is not None else None
        CtempLogsumexp = Ctlse[:k] if Ctlse is not None else None
        Bprobe = _probe_seq("current_log_probs", i)
        BrefProbe = _probe_seq("ref_log_probs", i)
        FtargetLogit = _probe_seq("target_logits", i)
        Flogsumexp = _probe_seq("logsumexp", i)
        FtargetIds = _probe_seq("target_ids", i)
        if Am is not None:
            accA.append(Am)
        if Bm is not None:
            accB.append(Bm)
        accC.append(Cm)
        row = {
            "seq": i,
            "n": k,
            "A": Am.detach().cpu() if Am is not None else None,
            "B": Bm.detach().cpu() if Bm is not None else None,
            "Bprobe": Bprobe.detach().cpu() if Bprobe is not None else None,
            "Bref": Brefm.detach().cpu() if Brefm is not None else None,
            "BrefProbe": BrefProbe.detach().cpu() if BrefProbe is not None else None,
            "C": Cm.detach().cpu(),
            "Ctemp": Ctm.detach().cpu() if Ctm is not None else None,
            "Cpack": None,
            "CpackTemp": None,
            "D": None,
            "tokens": Tm.detach().cpu() if Tm is not None else None,
            "FtargetIds": FtargetIds.detach().cpu().long() if FtargetIds is not None else None,
            "FtargetLogit": FtargetLogit.detach().cpu() if FtargetLogit is not None else None,
            "Flogsumexp": Flogsumexp.detach().cpu() if Flogsumexp is not None else None,
            "CtargetLogit": CtargetLogit.detach().cpu() if CtargetLogit is not None else None,
            "Clogsumexp": Clogsumexp.detach().cpu() if Clogsumexp is not None else None,
            "CtempTargetLogit": CtempTargetLogit.detach().cpu() if CtempTargetLogit is not None else None,
            "CtempLogsumexp": CtempLogsumexp.detach().cpu() if CtempLogsumexp is not None else None,
            "CpackTargetLogit": None,
            "CpackLogsumexp": None,
            "CpackTempTargetLogit": None,
            "CpackTempLogsumexp": None,
            "L": L,
            "resp_len": resp_len,
            "resp_valid": resp_valid,
            "prompt_len": prompt_len,
            "align_mode": align_mode,
            "resp_len_src": resp_len_src,
            "n_img_tok": n_img_tok,
            "temperature": _temp_for_seq(i),
        }
        seq_rows.append(row)
        pack_jobs.append(
            {
                "row": row,
                "ids": ids_i.detach().cpu(),
                "pos": pos_i.detach().cpu() if pos_i is not None else None,
                "resp_len": resp_len,
                "resp_valid": resp_valid,
                "prompt_len": prompt_len,
                "align_mode": align_mode,
                "mask": mask.detach().cpu() if torch.is_tensor(mask) else mask,
                "seq_pix": [p.detach().cpu() for p in seq_pix],
                "seq_grids": [g.detach().cpu() for g in seq_grids],
                "temperature": _temp_for_seq(i),
            }
        )
        if with_vllm:
            d_jobs.append(
                {
                    "row": row,
                    "ids": ids_i.detach().cpu(),
                    "resp_len": resp_len,
                    "resp_valid": resp_valid,
                    "prompt_len": prompt_len,
                    "align_mode": align_mode,
                    "mask": mask.detach().cpu() if torch.is_tensor(mask) else mask,
                    "seq_pix": [p.detach().cpu() for p in seq_pix],
                    "seq_grids": [g.detach().cpu() for g in seq_grids],
                    "raw_images": raw_images[i] if raw_images and i < len(raw_images) and raw_images[i] else None,
                    "temperature": _temp_for_seq(i),
                }
            )

    if not accC:
        _p("[ASYNC-BUNDLE] no comparable response tokens after alignment; check responses/response_mask lengths")
        return

    _run_packed_hf_recompute()

    if with_vllm and d_jobs:
        import gc

        ids_i = grid_i = pix_i = logits = resp_logits = kw = pos_i = target = frames = None
        _p("[ASYNC-BUNDLE] finished HF(C); releasing HF before loading fresh vLLM(D)")
        del hf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass

        llm = SamplingParams = None
        try:
            from vllm import LLM, SamplingParams

            raw_img_counts = [len(j["raw_images"] or []) for j in d_jobs if j["raw_images"]]
            grid_img_counts = [len(j["seq_grids"]) for j in d_jobs]
            max_imgs = max(raw_img_counts + grid_img_counts + [1])
            img_src_msg = "raw_images/pixel_values->PIL" if raw_img_counts else "reconstructed from pixel_values"
            _p(f"[ASYNC-BUNDLE] loading fresh vLLM(D), images={img_src_msg}, "
               f"logprobs_mode={logprobs_mode}, max_imgs={max_imgs}, gpu_mem={gpu_mem}")
            llm_kwargs = dict(
                model=model_path,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=gpu_mem,
                max_model_len=32768,
                limit_mm_per_prompt={"image": max_imgs},
                enforce_eager=True,
            )
            try:
                llm = LLM(logprobs_mode=logprobs_mode, **llm_kwargs)
            except TypeError:
                _p("[ASYNC-BUNDLE] this vLLM build does not accept logprobs_mode= on LLM(); using default")
                llm = LLM(**llm_kwargs)
        except Exception as e:  # noqa: BLE001
            import traceback

            _p(f"[ASYNC-BUNDLE] fresh vLLM(D) init failed: {e!r}; continuing with A/B/C only")
            traceback.print_exc()
            llm = None

        first_d = True
        if llm is not None:
            for job in d_jobs:
                row = job["row"]
                try:
                    ids_cpu = job["ids"].long().reshape(-1)
                    L = ids_cpu.numel()
                    if job["raw_images"]:
                        images = job["raw_images"]
                        image_src = "raw_images"
                    else:
                        images = (
                            _pixels_to_reconstructed_images(job["seq_pix"], job["seq_grids"], processor)
                            if job["seq_pix"]
                            else []
                        )
                        image_src = "pixel_values->PIL"
                    unexpanded_full, n_runs = _collapse_image_runs(ids_cpu.tolist(), image_token_id)
                    if n_runs != len(images):
                        _p(f"[ASYNC-BUNDLE] seq{row['seq']} fresh-vLLM WARN: image runs={n_runs} images={len(images)} "
                           f"src={image_src}; D may misalign")
                    temp_i = float(job["temperature"])
                    sp = SamplingParams(temperature=temp_i if temp_i > 0 else 1.0, max_tokens=1, prompt_logprobs=1)
                    if job["align_mode"] == "response-window":
                        resp_len = int(job["resp_len"])
                        prompt_len = int(job["prompt_len"])
                        response_ids = ids_cpu[prompt_len:L].tolist()
                        Dv0 = _vllm_seq_logprobs(llm, sp, unexpanded_full, images, response_ids, resp_len).float()
                        mask = job["mask"]
                        if not torch.is_tensor(mask):
                            mask = torch.ones(resp_len, dtype=torch.bool)
                        mask = mask.bool().reshape(-1)
                        n = min(Dv0.numel(), mask.numel())
                        Dv = Dv0[:n][mask[:n]]
                    else:
                        resp_valid = int(job["resp_valid"])
                        response_ids = ids_cpu[L - resp_valid:L].tolist()
                        Dv0 = _vllm_seq_logprobs(llm, sp, unexpanded_full, images, response_ids, resp_valid).float()
                        mask = job["mask"]
                        if torch.is_tensor(mask) and mask.numel() == Dv0.numel():
                            Dv = Dv0[mask.bool().reshape(-1)]
                        else:
                            Dv = Dv0[:resp_valid]

                    trim_names = (
                        "A", "B", "Bprobe", "Bref", "BrefProbe", "C", "Cpack", "Ctemp", "CpackTemp",
                        "tokens", "FtargetIds", "FtargetLogit", "Flogsumexp",
                        "CtargetLogit", "Clogsumexp", "CtempTargetLogit", "CtempLogsumexp",
                        "CpackTargetLogit", "CpackLogsumexp", "CpackTempTargetLogit", "CpackTempLogsumexp",
                    )
                    existing = [row[name] for name in trim_names if row.get(name) is not None]
                    k = min([Dv.numel()] + [x.numel() for x in existing])
                    if k == 0:
                        continue
                    for name in trim_names:
                        if row.get(name) is not None:
                            row[name] = row[name][:k]
                    row["D"] = Dv[:k].detach().cpu()
                    row["n"] = k
                    accD.append(row["D"])
                    if first_d:
                        first_d = False
                        _p(f"[ASYNC-BUNDLE] fresh-vLLM(D) image source: {image_src}; "
                           f"valid_logprobs={int(torch.isfinite(row['D']).sum())}/{row['D'].numel()}")
                except Exception as e:  # noqa: BLE001
                    import traceback

                    _p(f"[ASYNC-BUNDLE] seq{row['seq']} fresh-vLLM(D) failed: {e!r}")
                    traceback.print_exc()

    def _mean_abs_or_nan(x, y):
        s = _abs_stats(x, y)
        return s[0] if s is not None else float("nan")

    def _print_worst_pair(row, label, a_name, b_name, max_threshold=0.5, topk=5):
        x, y = row.get(a_name), row.get(b_name)
        if x is None or y is None:
            return
        n = min(x.numel(), y.numel())
        if n == 0:
            return
        d = (x[:n].to(torch.float64) - y[:n].to(torch.float64)).abs()
        finite = torch.isfinite(d).nonzero(as_tuple=True)[0]
        if finite.numel() == 0 or float(d[finite].max()) < max_threshold:
            return
        order = finite[torch.argsort(d[finite], descending=True)[:topk]]
        toks = row.get("tokens")
        _p(f"[ASYNC-BUNDLE] seq{row['seq']} worst {label} tokens "
           f"(pos | A B Bp Bref Brp C Cp Ct Cpt D | token):")
        for j in order.tolist():
            vals = []
            for name in ("A", "B", "Bprobe", "Bref", "BrefProbe", "C", "Cpack", "Ctemp", "CpackTemp", "D"):
                v = row.get(name)
                vals.append("   nan" if v is None or j >= v.numel() else f"{float(v[j]):+7.3f}")
            tok_id = int(toks[j]) if torch.is_tensor(toks) and j < toks.numel() else None
            tok = tokenizer.decode([tok_id]) if tok_id is not None and tokenizer is not None else str(tok_id)
            _p(f"    {j:4d} | {' '.join(vals)} | {tok!r}")

    def _print_worst_diag_pair(row, label, a_name, b_name, max_threshold=0.5, topk=5):
        x, y = row.get(a_name), row.get(b_name)
        if x is None or y is None:
            return
        n = min(x.numel(), y.numel())
        if n == 0:
            return
        d = (x[:n].to(torch.float64) - y[:n].to(torch.float64)).abs()
        finite = torch.isfinite(d).nonzero(as_tuple=True)[0]
        if finite.numel() == 0 or float(d[finite].max()) < max_threshold:
            return
        order = finite[torch.argsort(d[finite], descending=True)[:topk]]
        toks = row.get("tokens")
        _p(f"[ASYNC-BUNDLE] seq{row['seq']} worst {label} diagnostics (pos | {a_name} {b_name} | token):")
        for j in order.tolist():
            tok_id = int(toks[j]) if torch.is_tensor(toks) and j < toks.numel() else None
            tok = tokenizer.decode([tok_id]) if tok_id is not None and tokenizer is not None else str(tok_id)
            _p(f"    {j:4d} | {float(x[j]):+10.4f} {float(y[j]):+10.4f} | {tok!r}")

    for row in seq_rows:
        Am, Bm = row.get("A"), row.get("B")
        Bpm, Brpm = row.get("Bprobe"), row.get("BrefProbe")
        Brm, Cm = row.get("Bref"), row.get("C")
        Cpm, Ctm, Cptm, Dm = row.get("Cpack"), row.get("Ctemp"), row.get("CpackTemp"), row.get("D")
        dab = _mean_abs_or_nan(Am, Bm)
        dabp = _mean_abs_or_nan(Am, Bpm)
        dabr = _mean_abs_or_nan(Am, Brm)
        dabrp = _mean_abs_or_nan(Am, Brpm)
        dbbp = _mean_abs_or_nan(Bm, Bpm)
        dac = _mean_abs_or_nan(Am, Cm)
        dacp = _mean_abs_or_nan(Am, Cpm)
        dbbr = _mean_abs_or_nan(Bm, Brm)
        dbbrp = _mean_abs_or_nan(Bm, Brpm)
        dbpbrp = _mean_abs_or_nan(Bpm, Brpm)
        dbc = _mean_abs_or_nan(Bm, Cm)
        dbcp = _mean_abs_or_nan(Bm, Cpm)
        dbpc = _mean_abs_or_nan(Bpm, Cm)
        dbpcp = _mean_abs_or_nan(Bpm, Cpm)
        dbrc = _mean_abs_or_nan(Brm, Cm)
        dbrcp = _mean_abs_or_nan(Brm, Cpm)
        dbrpc = _mean_abs_or_nan(Brpm, Cm)
        dbrpcp = _mean_abs_or_nan(Brpm, Cpm)
        dccp = _mean_abs_or_nan(Cm, Cpm)
        dbct = _mean_abs_or_nan(Bm, Ctm)
        dbcpt = _mean_abs_or_nan(Bm, Cptm)
        dad = _mean_abs_or_nan(Am, Dm)
        dcd = _mean_abs_or_nan(Cm, Dm)
        dcpd = _mean_abs_or_nan(Cpm, Dm)
        kab = _rs_k3_mean(Am, Bm) if (Am is not None and Bm is not None) else float("nan")
        fids, toks = row.get("FtargetIds"), row.get("tokens")
        target_mismatch = "n/a"
        if torch.is_tensor(fids) and torch.is_tensor(toks):
            n_tid = min(fids.numel(), toks.numel())
            if n_tid:
                target_mismatch = f"{int((fids[:n_tid].long() != toks[:n_tid].long()).sum())}/{n_tid}"
        _p(f"[ASYNC-BUNDLE] seq{row['seq']} L={row['L']} resp_len={row['resp_len']} "
           f"resp_valid={row['resp_valid']} prompt_len={row['prompt_len']} "
           f"align={row['align_mode']}/{row['resp_len_src']} imgtok={row['n_img_tok']} "
           f"temp={row['temperature']:.4g} | "
           f"|A-B|dumpVLLM_vs_dumpFSDP={dab:.4f} |A-C|dumpVLLM_vs_freshHF={dac:.4f} "
           f"|A-Cp|dumpVLLM_vs_freshHF_PACK={dacp:.4f} "
           f"|B-C|dumpFSDP_vs_freshHF={dbc:.4f} |B-Cp|dumpFSDP_vs_freshHF_PACK={dbcp:.4f} "
           f"|C-Cp|singleHF_vs_packHF={dccp:.4f} "
           f"|B-Ctemp|dumpFSDP_vs_HF/temp={dbct:.4f} |B-Cptemp|dumpFSDP_vs_PACK/temp={dbcpt:.4f} "
           f"|A-D|dumpVLLM_vs_freshVLLM={dad:.4f} "
           f"|C-D|freshHF_vs_freshVLLM={dcd:.4f} |Cp-D|freshHF_PACK_vs_freshVLLM={dcpd:.4f}")
        if Bpm is not None or Brpm is not None:
            _p(f"[ASYNC-BUNDLE] seq{row['seq']} FSDP-exact-row-probe: target_id_mismatch={target_mismatch} "
               f"|B-Bprobe|slice_vs_exact={dbbp:.6f} |A-Bprobe|dumpVLLM_vs_exact={dabp:.4f} "
               f"|Bprobe-BrefProbe|kernel_vs_torch_ref={dbpbrp:.6f} "
               f"|Bprobe-C|exact_vs_freshHF={dbpc:.4f} |Bprobe-Cp|exact_vs_freshHF_PACK={dbpcp:.4f} "
               f"|BrefProbe-C|ref_vs_freshHF={dbrpc:.4f} |BrefProbe-Cp|ref_vs_freshHF_PACK={dbrpcp:.4f}")
        if Brm is not None:
            _p(f"[ASYNC-BUNDLE] seq{row['seq']} FSDP-logprob-check: "
               f"|B-Bref|kernel_vs_torch_ref={dbbr:.6f} |A-Bref|dumpVLLM_vs_ref={dabr:.4f} "
               f"|B-BrefProbe|slice_vs_exact_ref={dbbrp:.6f} |A-BrefProbe|dumpVLLM_vs_exact_ref={dabrp:.4f} "
               f"|Bref-C|torch_ref_vs_freshHF={dbrc:.4f} |Bref-Cp|torch_ref_vs_freshHF_PACK={dbrcp:.4f}")
        Ftgt, Flse = row.get("FtargetLogit"), row.get("Flogsumexp")
        Cttgt, Ctlse = row.get("CtempTargetLogit"), row.get("CtempLogsumexp")
        Cpttgt, Cptlse = row.get("CpackTempTargetLogit"), row.get("CpackTempLogsumexp")
        if Ftgt is not None and Flse is not None:
            _p(f"[ASYNC-BUNDLE] seq{row['seq']} FSDP-logits-vs-HF(scaled by temp): "
               f"|Ftarget-Ctarget|={_mean_abs_or_nan(Ftgt, Cttgt):.4f} "
               f"|Flse-Clse|={_mean_abs_or_nan(Flse, Ctlse):.4f} "
               f"|Ftarget-CpTarget|={_mean_abs_or_nan(Ftgt, Cpttgt):.4f} "
               f"|Flse-CpLse|={_mean_abs_or_nan(Flse, Cptlse):.4f}")
        _p(f"[ASYNC-BUNDLE] seq{row['seq']} RS-K3(A=vLLM||B=FSDP)={_fmt(kab)} "
           f"({'MASKED' if kab > 0.005 else 'kept'} @0.005)")
        _print_worst_pair(row, "|A-B|", "A", "B")
        _print_worst_pair(row, "|B-Bprobe|", "B", "Bprobe")
        _print_worst_pair(row, "|B-Bref|", "B", "Bref")
        _print_worst_pair(row, "|Bprobe-BrefProbe|", "Bprobe", "BrefProbe")
        _print_worst_pair(row, "|B-C|", "B", "C")
        _print_worst_pair(row, "|B-Cp|", "B", "Cpack")
        _print_worst_pair(row, "|Bprobe-C|", "Bprobe", "C")
        _print_worst_pair(row, "|Bprobe-Cp|", "Bprobe", "Cpack")
        _print_worst_pair(row, "|C-Cp|", "C", "Cpack")
        _print_worst_pair(row, "|Bref-C|", "Bref", "C")
        _print_worst_pair(row, "|Bref-Cp|", "Bref", "Cpack")
        _print_worst_pair(row, "|BrefProbe-C|", "BrefProbe", "C")
        _print_worst_pair(row, "|BrefProbe-Cp|", "BrefProbe", "Cpack")
        _print_worst_pair(row, "|A-D|", "A", "D")
        _print_worst_pair(row, "|C-D|", "C", "D")
        _print_worst_pair(row, "|Cp-D|", "Cpack", "D")
        _print_worst_diag_pair(row, "|Ftarget-Ctarget|", "FtargetLogit", "CtempTargetLogit")
        _print_worst_diag_pair(row, "|Flse-Clse|", "Flogsumexp", "CtempLogsumexp")
        _print_worst_diag_pair(row, "|Ftarget-CpTarget|", "FtargetLogit", "CpackTempTargetLogit")
        _print_worst_diag_pair(row, "|Flse-CpLse|", "Flogsumexp", "CpackTempLogsumexp")

    _p("\n[ASYNC-BUNDLE] ===== AGGREGATE over all response tokens =====")

    def _row_pair(a, bname):
        xs, ys = [], []
        for row in seq_rows:
            x, y = row.get(a), row.get(bname)
            if x is None or y is None:
                continue
            n = min(x.numel(), y.numel())
            if n:
                xs.append(x[:n])
                ys.append(y[:n])
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    def _print_pair(label, a, bname):
        x, y = _row_pair(a, bname)
        if x is not None:
            _p(f"  {label}: {_fmt_abs(x, y)}")

    _print_pair("|A-B| dump-vLLM vs dump-FSDP       ", "A", "B")
    _print_pair("|A-Bp| dump-vLLM vs FSDP exact-row ", "A", "Bprobe")
    _print_pair("|A-Br| dump-vLLM vs FSDP torch-ref ", "A", "Bref")
    _print_pair("|A-Brp| dump-vLLM vs exact ref     ", "A", "BrefProbe")
    _print_pair("|A-C| dump-vLLM vs fresh-HF/FSDP    ", "A", "C")
    _print_pair("|A-Cp| dump-vLLM vs fresh-HF PACK   ", "A", "Cpack")
    _print_pair("|B-Bp| FSDP sliced vs exact-row     ", "B", "Bprobe")
    _print_pair("|B-Br| FSDP kernel vs torch-ref     ", "B", "Bref")
    _print_pair("|B-Brp| FSDP sliced vs exact ref    ", "B", "BrefProbe")
    _print_pair("|Bp-Brp| exact kernel vs exact ref  ", "Bprobe", "BrefProbe")
    _print_pair("|B-C| dump-FSDP vs fresh-HF (sanity)", "B", "C")
    _print_pair("|B-Cp| dump-FSDP vs fresh-HF PACK   ", "B", "Cpack")
    _print_pair("|Bp-C| exact FSDP vs fresh-HF       ", "Bprobe", "C")
    _print_pair("|Bp-Cp| exact FSDP vs fresh-HF PACK ", "Bprobe", "Cpack")
    _print_pair("|Br-C| FSDP torch-ref vs fresh-HF   ", "Bref", "C")
    _print_pair("|Br-Cp| FSDP torch-ref vs HF PACK   ", "Bref", "Cpack")
    _print_pair("|Brp-C| exact ref vs fresh-HF       ", "BrefProbe", "C")
    _print_pair("|Brp-Cp| exact ref vs fresh-HF PACK ", "BrefProbe", "Cpack")
    _print_pair("|C-Cp| fresh-HF single vs HF PACK   ", "C", "Cpack")
    _print_pair("|B-Ct| dump-FSDP vs fresh-HF/temp   ", "B", "Ctemp")
    _print_pair("|B-Cpt| dump-FSDP vs HF PACK/temp   ", "B", "CpackTemp")
    _print_pair("|A-D| dump-vLLM vs fresh-vLLM       ", "A", "D")
    _print_pair("|B-D| dump-FSDP vs fresh-vLLM       ", "B", "D")
    _print_pair("|C-D| fresh-HF vs fresh-vLLM        ", "C", "D")
    _print_pair("|Cp-D| fresh-HF PACK vs fresh-vLLM  ", "Cpack", "D")
    _p("\n[ASYNC-BUNDLE] ===== aggregate FSDP logits diagnostics (FSDP logits are temp-scaled) =====")
    _print_pair("|Ftarget-Ctarget| FSDP target-logit vs HF     ", "FtargetLogit", "CtempTargetLogit")
    _print_pair("|Flse-Clse| FSDP logsumexp vs HF              ", "Flogsumexp", "CtempLogsumexp")
    _print_pair("|Ftarget-CpTarget| FSDP target-logit vs HF PACK", "FtargetLogit", "CpackTempTargetLogit")
    _print_pair("|Flse-CpLse| FSDP logsumexp vs HF PACK         ", "Flogsumexp", "CpackTempLogsumexp")
    _print_k3_matrix("ASYNC-BUNDLE", seq_rows)
    if not accD:
        _p("[ASYNC-BUNDLE] D=fresh-vLLM is unavailable because vLLM init/replay failed; A/B/C are still valid.")
    _p("  READ: A=dump vLLM, B=dump FSDP, C=fresh HF single-seq, Cp=fresh HF packed-varlen, D=fresh vLLM.")
    _p("        |B-Bp| large => bench slicing/alignment or dumped fsdp_log_probs frame is wrong; trust Bp exact-row.")
    _p("        |Bp-Brp| large while Ftarget/Flse match HF => FSDP logprobs_from_logits/gather path is wrong.")
    _p("        Ftarget/Flse differ from C/Cp => FSDP forward logits/state differ from fresh HF, before logprob math.")
    _p("        |C-Cp| nonzero => fresh packed replay differs from single replay; otherwise normal packing is clean.")
    _p("        |A-D| small => dumped vLLM is reproducible; |C-D|/|Cp-D| are clean HF-vs-vLLM gaps.")


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
    ap.add_argument("--async-bundle", required=True,
                    help="Probe .pt with dump-vLLM, dump-FSDP, tokens, position_ids, pixel_values, image_grid_thw.")
    ap.add_argument("--attn", default="flash_attention_2", help="HF attn_implementation (match verl).")
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM gpu_memory_utilization.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    _p("\n################## ABCD+Cp BENCH (A=dump vLLM / B=dump FSDP / C=fresh HF single / Cp=fresh HF packed / D=fresh vLLM) ##################")
    _run_async_bundle(
        args.async_bundle,
        args.model,
        None,
        processor,
        tokenizer,
        device,
        feed_pos=True,
        with_vllm=True,
        attn=args.attn,
        gpu_mem=args.gpu_mem,
        max_seq=0,
    )


if __name__ == "__main__":
    main()
