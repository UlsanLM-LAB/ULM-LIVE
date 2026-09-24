# ULM Live v2 TTS pivot

Date: 2026-09-24.

## Decision

Use the official `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` model for Korean speech with preset speaker `Sohee`. ULM-1.7B remains the text generator. The Mimi Talker training work is frozen as research; its checkpoints, datasets, caches, and logs stay on EC2. This path avoids depending on the research Talker's low held-out autoregressive codec accuracy for understandable speech.

## Verified upstream facts

- The [Qwen3-TTS release table](https://github.com/QwenLM/Qwen3-TTS#released-models-description-and-download) lists both 1.7B and 0.6B CustomVoice models, Korean support, and streaming capability. It lists the 1.7B Base model for three-second voice cloning and fine-tuning.
- The [official Python example](https://github.com/QwenLM/Qwen3-TTS#custom-voice-generate) uses `Qwen3TTSModel.from_pretrained` and `generate_custom_voice(text=..., language=..., speaker=...)`. It names `Sohee` as the native Korean preset speaker. This implementation uses the 1.7B model, bfloat16, and PyTorch SDPA. FlashAttention is optional upstream.
- The [model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) identifies Apache-2.0 licensing. The [upstream code](https://github.com/QwenLM/Qwen3-TTS) also carries Apache-2.0.
- The [voice clone example](https://github.com/QwenLM/Qwen3-TTS#voice-clone) requires a Base model and reference audio; this is a separate model and is outside the initial service. The [fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md) currently documents single-speaker fine-tuning of Base models, with `audio`, `text`, and `ref_audio` JSONL fields.
- Upstream says its architecture supports streaming. The current Python `generate_custom_voice` example returns a completed waveform. Native chunk delivery is therefore deferred until the non-streaming path has demonstrated intelligible Korean and an official usable streaming API is verified on this install.

## Evaluation gate

First synthesize 20 Korean, Ulsan, and conversational sentences and retain WAV plus per-sample metadata on EC2. Check sample rate, duration, finite non-silent signal, and latency. No Korean ASR model was already cached, so this run cannot report CER or certify intelligibility without listening. Only after the structural zero-shot run succeeds, execute 20 ULM-to-TTS prompts. Voice cloning and fine-tuning are separate later gates and are not necessary for v2 speech delivery.

The `Sohee` model is a standard Korean preset, not a proven Ulsan-accent model. Accent accuracy needs native-speaker evaluation. The runtime serializes GPU requests, loads models once at startup, and serves on loopback by default in the documented command.
