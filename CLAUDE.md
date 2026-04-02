# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project generating synthetic accented English speech data using multilingual TTS models, then building accent-aware STT routing for vLLM. University of Chicago, Junchen Jiang Research Group (NSF NAIRR funded).

**Core insight**: TTS models trained on non-English languages naturally produce accented English, providing free ground-truth transcriptions at scale.

**Four research phases**:
1. Data generation pipeline (current phase)
2. Accent classifier training
3. Accent-adaptive STT fine-tuning (Whisper + LoRA)
4. vLLM semantic routing integration

## Cluster & GPU Operations

Infrastructure is NERC MGHPCC OpenShift with H100 GPUs at ~$3/hr. **Always scale down after use.**

```bash
# Login
oc login --token=<token> --server=https://k8s.nerc.mghpcc.org:6443

# Scale up a model (takes 3-8 min to load)
oc scale deployment qwen3-tts --replicas=1

# Port-forward for local API access
oc port-forward svc/qwen3-tts 8091:8000 &
oc port-forward svc/voxtral-tts 8092:8000 &
oc port-forward svc/fish-speech 8093:8000 &

# Verify model is ready
curl http://localhost:8091/v1/audio/voices

# ALWAYS scale down when done
oc scale deployment qwen3-tts --replicas=0
oc scale deployment qwen3-tts-base --replicas=0
oc scale deployment fish-speech --replicas=0
oc scale deployment voxtral-tts --replicas=0
```

## Running Data Generation

```bash
# Pilot/single-model generation
python src/data_generation/generate_accent_samples.py --api http://localhost:8091

# Production multi-model generation
python src/data_generation/generate_accent_dataset.py \
  --qwen-api http://localhost:8091 \
  --voxtral-api http://localhost:8092 \
  --output-dir data/synthetic
```

## Data & Experiment Conventions

**Run naming**: `{model_short}_{YYYYMMDD}_{NNN}` (e.g., `qwen3tts_20260330_001`, `voxtral_20260330_001`)

Each run directory contains:
- `run_config.json` — model metadata, voices, scenarios, hardware config
- `manifest.json` — per-sample metadata (voice, accent, text, file path, generation timing)
- `annotations.json` — optional human feedback on sample quality

Audio layout: `data/synthetic/{run_id}/{accent}/{voice}/{sentence_id}.wav`

Reference benchmark audio: `data/reference_voices/` (Speech Accent Archive samples)

**When adding new experiments**: Always create a new run_id, write manifest.json, and document findings in `progress.txt` with the session date.

## Architecture

**API interface**: vLLM-Omni exposes OpenAI-compatible `/v1/audio/speech` (POST with `input`, `voice`, `response_format`, optional `instructions`). Timeout is 120-180s per generation.

**Kubernetes manifests**: `k8s/*.yaml` — separate deployments per TTS model, all use PVC for HF model cache, H100 GPU tolerations, resource limits defined.

## Model Status

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3-TTS CustomVoice | ✅ Best | Use Chinese instruction: `"你在跟餐馆打电话，速度快一点，带着chinglish的口音"` |
| Qwen3-TTS Base | ⚠️ Partial | Voice cloning works, but reference_text fragments leak into output |
| Voxtral TTS | ⚠️ Partial | Acceptable accent quality; speed > 1.0 causes artifacts |
| Fish Speech S2 Pro | ❌ Poor | Clones timbre only, not accent; inline tags ineffective |
| CosyVoice3 | ❌ Fails | Deployment fails — empty config.json on HuggingFace |

## Commit Convention

Follow conventional commits: `type(scope): description`

Types: `feat`, `fix`, `docs`, `exp`, `data`, `k8s`, `refactor`

Examples: `exp: cross-lingual accent test`, `data: commit qwen3tts_20260330_001 samples`
