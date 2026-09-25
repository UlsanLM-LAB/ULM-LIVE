#!/usr/bin/env python3
"""Smoke adaptation pipeline for Qwen3-TTS Ulsan dialect fine-tuning.

Execution steps:
1. Load pretrained Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice on CUDA.
2. Synthesize baseline audio for 12 standardized evaluation prompts.
3. Pre-encode training audio codes to NVMe cache for high-throughput training.
4. Parameter-efficient fine-tuning on talker top decoder layers and codec head.
5. Train with AdamW, linear warmup + cosine decay, and gradient accumulation.
6. Synthesize adapted audio for the exact same 12 evaluation prompts.
7. Compute automated sanity metrics (duration, RMS, clipping, silence).
8. Save artifacts to outputs/tts-smoke-v1/.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel


EVAL_PROMPTS = [
    # A. Everyday conversation (4)
    {"id": "01", "category": "everyday", "text": "오늘 학교 끝나고 뭐 할 거야?"},
    {"id": "02", "category": "everyday", "text": "밥은 먹었나?"},
    {"id": "03", "category": "everyday", "text": "어디 가노?"},
    {"id": "04", "category": "everyday", "text": "빨리 와라, 늦겠다."},
    # B. Ulsan / Gyeongsang dialect expressions (4)
    {"id": "05", "category": "dialect", "text": "단디 해라."},
    {"id": "06", "category": "dialect", "text": "퍼뜩 온나."},
    {"id": "07", "category": "dialect", "text": "맞나?"},
    {"id": "08", "category": "dialect", "text": "와 이라노?"},
    # C. Standard Korean sentences (4)
    {"id": "09", "category": "standard", "text": "오늘 날씨가 정말 좋다."},
    {"id": "10", "category": "standard", "text": "지금 뭐 하고 있어?"},
    {"id": "11", "category": "standard", "text": "조심해서 들어가."},
    {"id": "12", "category": "standard", "text": "저녁 먹고 같이 산책하자."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke adaptation for Qwen3-TTS Ulsan dialect.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--train-manifest", type=str, default="data/ulm-live-tts-v2/smoke/smoke_train.jsonl")
    parser.add_argument("--val-manifest", type=str, default="data/ulm-live-tts-v2/smoke/smoke_validation.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/tts-smoke-v1")
    parser.add_argument("--cache-dir", type=str, default="/opt/dlami/nvme/smoke_code_cache")
    parser.add_argument("--speaker", type=str, default="Sohee")
    parser.add_argument("--language", type=str, default="Korean")
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=70)
    parser.add_argument("--max-runtime-min", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_audio_sanity(wav: np.ndarray, sr: int) -> dict[str, Any]:
    duration = len(wav) / float(sr)
    rms = float(np.sqrt(np.mean(np.square(wav))))
    peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    clipped = float(np.mean(np.abs(wav) >= 0.999))
    # Silence ratio: frame energy < -40 dB relative to peak
    frame_len = int(sr * 0.02)
    hop = int(sr * 0.01)
    if len(wav) >= frame_len:
        frames = [
            wav[i : i + frame_len]
            for i in range(0, len(wav) - frame_len, hop)
        ]
        frame_rms = [np.sqrt(np.mean(f ** 2)) for f in frames]
        threshold = max(peak * 0.01, 1e-4)
        silence_ratio = float(np.mean([r < threshold for r in frame_rms]))
    else:
        silence_ratio = 0.0

    return {
        "duration_seconds": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "clipped_ratio": round(clipped, 6),
        "silence_ratio": round(silence_ratio, 4),
    }


def generate_evaluation_set(
    tts: Qwen3TTSModel,
    prompts: list[dict[str, str]],
    output_dir: Path,
    speaker: str,
    language: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Generating {len(prompts)} evaluation samples to {output_dir}...")

    for item in prompts:
        pid = item["id"]
        text = item["text"]
        cat = item["category"]
        wav_file = output_dir / f"{pid}.wav"

        t0 = time.time()
        with torch.no_grad():
            wavs, rate = tts.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
            )
        latency = time.time() - t0

        audio = np.asarray(wavs[0], dtype=np.float32)
        sf.write(wav_file, audio, rate, subtype="PCM_16")

        sanity = compute_audio_sanity(audio, rate)
        res = {
            "id": pid,
            "category": cat,
            "text": text,
            "file": f"{pid}.wav",
            "sample_rate": rate,
            "latency_seconds": round(latency, 3),
            **sanity,
        }
        results.append(res)
        print(f"  [{pid}] ({cat}) {text} -> {sanity['duration_seconds']}s (RMS: {sanity['rms']})")

    return results


def build_teacher_forced_inputs(
    tts: Qwen3TTSModel,
    text: str,
    codes: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs input_embeds and target labels for a single sample."""
    model = tts.model
    talker = model.talker
    processor = tts.processor

    T = codes.shape[0]

    # Tokenize assistant text
    assistant_text = tts._build_assistant_text(text)
    input_ids = processor(text=assistant_text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        spk_id = model.config.talker_config.spk_id["sohee"]
        speaker_embed = talker.get_input_embeddings()(torch.tensor([spk_id], device=device))
        language_id = model.config.talker_config.codec_language_id["korean"]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                    device=device,
                )
            )
        ).chunk(3, dim=1)

        codec_prefill_list = [[
            model.config.talker_config.codec_think_id,
            model.config.talker_config.codec_think_bos_id,
            language_id,
            model.config.talker_config.codec_think_eos_id,
        ]]

        codec_input_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill_list, device=device))
        codec_input_1 = talker.get_input_embeddings()(
            torch.tensor([[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]], device=device)
        )

        codec_input_emebdding = torch.cat([codec_input_0, speaker_embed.view(1, 1, -1), codec_input_1], dim=1)

        _talker_role = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))
        _talker_pref = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input_emebdding[:, :-1]
        talker_prefix = torch.cat((_talker_role, _talker_pref), dim=1)

        text_embed = torch.cat((
            talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:-5])),
            tts_eos_embed,
        ), dim=1) + talker.get_input_embeddings()(
            torch.tensor([[model.config.talker_config.codec_pad_id] * (input_ids[:, 3:-5].shape[1] + 1)], device=device)
        )
        bos_embed = tts_pad_embed + talker.get_input_embeddings()(
            torch.tensor([[model.config.talker_config.codec_bos_id]], device=device)
        )

        prefix_embeds = torch.cat([talker_prefix, text_embed, bos_embed], dim=1)
        prefix_len = prefix_embeds.shape[1]

        # Audio frame embeddings
        frame_hiddens = [talker.get_input_embeddings()(codes[:, 0:1])]
        for i in range(15):
            frame_hiddens.append(talker.code_predictor.get_input_embeddings()[i](codes[:, i+1:i+2]))
        frame_embeds = torch.cat(frame_hiddens, dim=1).sum(dim=1, keepdim=True).transpose(0, 1) + tts_pad_embed

    full_inputs = torch.cat([prefix_embeds, frame_embeds], dim=1)
    labels = torch.full((1, full_inputs.shape[1]), -100, dtype=torch.long, device=device)
    labels[0, prefix_len - 1 : prefix_len + T - 1] = codes[:, 0]
    labels[0, prefix_len + T - 1] = model.config.talker_config.codec_eos_token_id

    return full_inputs, labels


def cache_codes_if_needed(
    tokenizer: Any,
    manifest_path: str,
    cache_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"Ensuring audio codec token cache in {cache_dir} for {len(items)} samples...")
    cached_items = []
    t0 = time.time()
    cache_count = 0

    for idx, item in enumerate(items):
        uid = item["id"]
        cache_file = cache_dir / f"{uid}.pt"
        if not cache_file.is_file():
            audio_path = item["rel_audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(audio_path)
            with torch.no_grad():
                enc = tokenizer.encode(audio_path)
                codes = enc.audio_codes[0].cpu().to(torch.int16)
            torch.save(codes, cache_file)
            cache_count += 1
        cached_items.append({
            "id": uid,
            "text": item["text"],
            "cache_file": str(cache_file),
            "speaker_id": item["speaker_id"],
            "duration": item["duration"],
        })
        if (idx + 1) % 500 == 0 or (idx + 1) == len(items):
            print(f"  Checked {idx + 1}/{len(items)} (newly encoded: {cache_count})...")

    print(f"Cache verification complete ({time.time() - t0:.1f}s). Total cached samples: {len(cached_items)}")
    return cached_items


def evaluate_validation_loss(
    tts: Qwen3TTSModel,
    val_items: list[dict[str, Any]],
    device: torch.device,
    max_eval_samples: int = 50,
) -> float:
    tts.model.talker.eval()
    losses = []
    sample_pool = list(val_items)
    random.shuffle(sample_pool)
    eval_subset = sample_pool[:max_eval_samples]

    with torch.no_grad():
        for item in eval_subset:
            codes = torch.load(item["cache_file"], map_location=device).long()
            inputs_embeds, labels = build_teacher_forced_inputs(tts, item["text"], codes, device)
            out = tts.model.talker(inputs_embeds=inputs_embeds, labels=labels)
            if out.loss is not None and torch.isfinite(out.loss):
                losses.append(out.loss.item())

    tts.model.talker.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    start_timestamp = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print("=================================================================")
    print("  ULM-LIVE: Qwen3-TTS Ulsan Accent Smoke Fine-Tuning Pipeline   ")
    print("=================================================================")
    print(f"Base model:          {args.model_name}")
    print(f"Train manifest:      {args.train_manifest}")
    print(f"Validation manifest: {args.val_manifest}")
    print(f"Output directory:    {output_dir}")
    print(f"Max steps:           {args.max_steps}")
    print(f"Gradient accum:      {args.grad_accum}")
    print(f"Learning rate:       {args.lr}")
    print(f"Max runtime limit:   {args.max_runtime_min} minutes")
    print(f"Speaker:             {args.speaker}")
    print(f"Language:            {args.language}")
    print("=================================================================")

    # Write prompts.json
    prompts_file = output_dir / "prompts.json"
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(EVAL_PROMPTS, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Pretrained Qwen3-TTS
    print("\n[1/5] Loading pretrained Qwen3-TTS CustomVoice...")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = tts.model
    talker = model.talker
    tokenizer = model.speech_tokenizer

    # 2. Generate Base Evaluation Samples
    print("\n[2/5] Generating Base Model Evaluation Audio...")
    base_dir = output_dir / "base"
    base_metrics = generate_evaluation_set(
        tts, EVAL_PROMPTS, base_dir, speaker=args.speaker, language=args.language
    )

    # 3. Cache Audio Codes for Training & Validation
    print("\n[3/5] Pre-encoding Audio Codes on NVMe Scratch Storage...")
    train_cached = cache_codes_if_needed(tokenizer, args.train_manifest, cache_dir / "train", device)
    val_cached = cache_codes_if_needed(tokenizer, args.val_manifest, cache_dir / "val", device)

    # 4. Prepare Model for Parameter-Efficient Adaptation
    print("\n[4/5] Configuring Parameter-Efficient Fine-Tuning...")
    # Clone parameter data to ensure autograd leaf tensors
    for p in talker.parameters():
        p.data = p.data.clone()

    talker.requires_grad_(False)

    # Adapt top 2 transformer layers and codec_head
    for p in talker.model.layers[-2:].parameters():
        p.requires_grad = True
    for p in talker.codec_head.parameters():
        p.requires_grad = True

    trainable_params = [p for p in talker.parameters() if p.requires_grad]
    trainable_numel = sum(p.numel() for p in trainable_params)
    total_numel = sum(p.numel() for p in talker.parameters())
    print(f"Trainable parameters: {trainable_numel/1e6:.2f}M / {total_numel/1e6:.2f}M ({trainable_numel/total_numel*100:.2f}%)")

    talker.train()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    def get_lr_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    # 5. Training Loop
    print("\n[5/5] Commencing Smoke Adaptation Training...")
    torch.cuda.reset_peak_memory_stats(device)

    step = 0
    epoch = 0
    accum_loss = 0.0
    loss_history = []
    val_loss_history = []

    # Initial validation loss
    init_val_loss = evaluate_validation_loss(tts, val_cached, device, max_eval_samples=30)
    print(f"Initial Validation Loss (Pre-train): {init_val_loss:.4f}")
    val_loss_history.append({"step": 0, "val_loss": round(init_val_loss, 4)})

    train_indices = list(range(len(train_cached)))
    sample_cursor = 0
    random.shuffle(train_indices)

    training_start_time = time.time()
    optimizer.zero_grad()

    while step < args.max_steps:
        # Check runtime limit
        elapsed_min = (time.time() - training_start_time) / 60.0
        if elapsed_min >= args.max_runtime_min:
            print(f"Reached maximum runtime limit ({args.max_runtime_min:.1f} min). Terminating training loop.")
            break

        # Accumulation window
        for accum_idx in range(args.grad_accum):
            if sample_cursor >= len(train_indices):
                epoch += 1
                random.shuffle(train_indices)
                sample_cursor = 0

            item = train_cached[train_indices[sample_cursor]]
            sample_cursor += 1

            codes = torch.load(item["cache_file"], map_location=device).long()
            inputs_embeds, labels = build_teacher_forced_inputs(tts, item["text"], codes, device)

            out = talker(inputs_embeds=inputs_embeds, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        current_lr = scheduler.get_last_lr()[0]
        step_loss = accum_loss
        accum_loss = 0.0
        loss_history.append({"step": step, "loss": round(step_loss, 4), "lr": round(current_lr, 7)})

        if step % 10 == 0 or step == 1:
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"Step {step:3d}/{args.max_steps} | Epoch {epoch} | Loss: {step_loss:.4f} | LR: {current_lr:.2e} | VRAM: {peak_vram_gb:.2f}GB | Elapsed: {elapsed_min:.1f}m")

        if step % args.eval_steps == 0 or step == args.max_steps:
            val_loss = evaluate_validation_loss(tts, val_cached, device, max_eval_samples=30)
            val_loss_history.append({"step": step, "val_loss": round(val_loss, 4)})
            print(f"  >>> [Eval Step {step}] Validation Loss: {val_loss:.4f} (Base: {init_val_loss:.4f})")

    total_training_time_sec = time.time() - training_start_time
    total_training_time_min = total_training_time_sec / 60.0
    final_loss = loss_history[-1]["loss"] if loss_history else float("nan")
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("\n=== Training Completed ===")
    print(f"Total Steps:    {step}")
    print(f"Training Time:  {total_training_time_min:.2f} min ({total_training_time_sec:.1f}s)")
    print(f"Final Loss:     {final_loss:.4f}")
    print(f"Peak VRAM:      {peak_vram_gb:.2f} GB")

    # Save adapted weights (only trainable adapted layers)
    adapted_weights_path = output_dir / "adapted_weights.pt"
    adapted_state_dict = {
        k: v.cpu()
        for k, v in talker.state_dict().items()
        if any(f"model.layers.{i}." in k for i in [26, 27]) or "codec_head." in k
    }
    torch.save(adapted_state_dict, adapted_weights_path)
    print(f"Saved adapted weights to {adapted_weights_path} ({len(adapted_state_dict)} keys, {sum(p.numel() for p in adapted_state_dict.values())/1e6:.2f}M params)")

    # 6. Generate Adapted Evaluation Audio
    print("\nGenerating Adapted Model Evaluation Audio...")
    adapted_dir = output_dir / "adapted"
    talker.eval()
    adapted_metrics = generate_evaluation_set(
        tts, EVAL_PROMPTS, adapted_dir, speaker=args.speaker, language=args.language
    )

    # 7. Compile Final Metrics and Config
    config_dict = {
        "model_name": args.model_name,
        "speaker": args.speaker,
        "language": args.language,
        "trainable_parameters": trainable_numel,
        "total_parameters": total_numel,
        "trainable_ratio": round(trainable_numel / total_numel, 4),
        "adapted_modules": ["model.layers.26", "model.layers.27", "codec_head"],
        "max_steps": args.max_steps,
        "effective_batch_size": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    metrics_dict = {
        "training": {
            "status": "COMPLETED",
            "completed_steps": step,
            "runtime_seconds": round(total_training_time_sec, 2),
            "runtime_minutes": round(total_training_time_min, 2),
            "initial_val_loss": round(init_val_loss, 4),
            "final_train_loss": round(final_loss, 4),
            "final_val_loss": val_loss_history[-1]["val_loss"] if val_loss_history else round(init_val_loss, 4),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "loss_history": loss_history,
            "validation_history": val_loss_history,
        },
        "evaluation": {
            "prompt_count": len(EVAL_PROMPTS),
            "base_samples": base_metrics,
            "adapted_samples": adapted_metrics,
        },
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    total_pipeline_time_sec = time.time() - start_timestamp
    print(f"\nAll smoke adaptation tasks finished successfully in {total_pipeline_time_sec/60.0:.2f} min!")


if __name__ == "__main__":
    main()
