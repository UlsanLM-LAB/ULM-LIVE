# ULM Live v2 TTS migration

Date: 2026-09-24. EC2: `i-0f732bf7d1cc409b4`, Seoul region.

## What changed

The v2 path uses the existing ULM-1.7B merged checkpoint for Korean response text, then the official Qwen3-TTS 1.7B CustomVoice model with `Sohee` for 24 kHz WAV output. The service loads both at startup and serializes GPU requests. `/v1/audio/speech`, `/v1/chat/speech`, and `/health` are available on a loopback-bound Uvicorn worker. The old Mimi Talker remains frozen research; Branch B was stopped by SIGINT after the total-step-11000 evaluation and its checkpoint, history, and logs were preserved.

The current root EBS volume `vol-07580a7ba8e092248` was expanded from 150 to 180 GiB on user approval. `growpart` and `resize2fs` increased the mounted ext4 filesystem to 175 GiB usable, with 33 GiB free before TTS installation. No existing checkpoint, dataset, cache, or WAV was deleted. Qwen and Python packages were installed in a separate persistent `.venv-tts`; Hugging Face weights remain on the root EBS cache.

## Zero-shot evaluation

All 20 selected Korean/Ulsan/conversational sentences generated WAVs on EC2 in `outputs/ulm-live-v2/zero-shot`. The per-sample text and measurements are in `metadata.jsonl` there, with a small copy at [zero-shot metadata](artifacts/zero-shot-metadata.jsonl). All were mono 24 kHz, finite, and non-silent. Duration ranged **2.08–4.80 s**, RMS **0.047302–0.126630**, and median generation latency **5.643 s** on one L40S. No Korean ASR model was cached, so CER and actual intelligibility were not measured. The preset is standard Korean; Ulsan accent fidelity is unverified.

## End-to-end evaluation

The first deterministic ULM attempt showed repetition and off-topic responses, so it was preserved as a partial diagnostic and stopped. A text-only comparison in both Transformers 4.57 and the checkpoint's original Transformers 5.17 environment reproduced off-topic output. The service now uses the ULM repository's mild-dialect system instruction and recommended sampling (`temperature=0.7`, `top_p=0.9`, `top_k=20`), with a 64-token limit and explicit attention mask.

The fresh 20-prompt run generated **20/20 WAVs** in `outputs/ulm-live-v2/e2e-sampled`, each 24 kHz, finite, and non-silent. Duration ranged **1.84–22.24 s**, RMS **0.022646–0.130609**, and median combined text-plus-speech latency was **8.952 s**. The `metadata.jsonl` preserves every prompt, spoken text, and measurement, with a small copy at [end-to-end metadata](artifacts/e2e-metadata.jsonl). The output text is **not production reliable**: prompt 3 asked about a pleasant-weather activity and answered about a computer program; prompt 8 repeated `친구야` to the token limit, ending in a replacement character. This is a ULM checkpoint text-quality failure. WAV integrity does not establish that the spoken answer is useful or factually correct.

## Verification and limits

The EC2 image has a system CUDA library path that conflicts with the TTS environment's cuDNN. Putting `.venv-tts/lib/python3.12/site-packages/nvidia/cudnn/lib` first in `LD_LIBRARY_PATH` resolved the first-sample decoder failure. The ULM tokenizer was saved by Transformers 5, while Qwen TTS installs Transformers 4.57; passing `extra_special_tokens={}` to the tokenizer loader handles the differing config type without changing the checkpoint.

Local repository tests: 85 passed, 1 skipped (TTS dependencies absent locally), 4 deselected. The TTS endpoint test passed on the EC2 TTS environment using a fake model runtime. Native streaming, voice cloning, fine-tuning, and listening or ASR-based quality certification are outside the validated v2 path. All WAVs and large files remain on EC2.

The real single-worker API also passed a loopback smoke test: `/health` returned ready; `/v1/audio/speech` returned HTTP 200 and a 3.60 s, 24 kHz WAV; `/v1/chat/speech` returned HTTP 200 and a 9.60 s, 24 kHz WAV with decodable ULM text in `X-ULM-Text`. The API test WAVs remain under `outputs/ulm-live-v2` on EC2. The smoke server was shut down cleanly after verification.

Ruff passed on the new and archived v4 source and tests. Repository-wide Ruff still reports 85 pre-existing findings in older files, outside this migration. Compileall passed on source, scripts, and tests.
