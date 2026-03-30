# AGENTS.md — Multi-Modal Semantic Routing for vLLM

## Quick Start (Every Session)
1. `cat progress.txt | tail -40` — last session state
2. `oc whoami 2>/dev/null || source .env && eval "$LOGIN"` — ensure cluster access
3. `oc get pods -l app=qwen3-tts` — check model serving status
4. Review `data/synthetic/` for existing samples

## Project Overview
Accent-aware speech recognition via TTS-synthesized training data. Use multilingual TTS models to generate accented English speech, then train classifiers and accent-adaptive STT models. NAIRR Core AI Track, UChicago.

## Architecture
- **Cluster**: NERC MGHPCC OpenShift (H100 GPUs, 8 GPU quota)
- **Model Serving**: vLLM-Omni on OpenShift (k8s/ manifests)
- **Data Pipeline**: Python scripts in `src/data_generation/`
- **Evaluation**: `src/accent_classifier/` (TBD), `experiments/`

## Data Generation Rules
- **Folder structure**: `data/synthetic/{run_id}/` per generation run
- **Every run MUST have**: `run_config.json` (model, voice, params, timestamp)
- **Every sample MUST be logged**: in `manifest.json` with full generation params
- **Human feedback**: Record in `annotations.json` per run (voice, sentence_id, rating, notes)
- **Never overwrite existing runs** — create new run_id
- **run_id format**: `{model_short}_{YYYYMMDD}_{NNN}` (e.g., `qwen3tts_20260330_001`)

## TTS Model Notes
- **Qwen3-TTS CustomVoice**: 9 preset voices. Good accent on serena, dylan, ono_anna, sohee, aiden. vivian/uncle_fu/eric/ryan too dramatic — need parameter tuning (top_p, temperature).
- **Voxtral TTS**: 4B params, 20 preset voices, 9 languages, zero-shot voice clone. Cross-lingual = accent generation. Deploy via vLLM-Omni same as Qwen3.
- Always test with small batch first, get human feedback, then scale.

## Coding Rules
- All code in `src/`, experiments in `experiments/`, k8s in `k8s/`
- Git identity: `Bo Wen <t-9bwen@uchicago.edu>` (repo-local config)
- SSH remote: `uchicago-github` alias
- **NO mentions of Panbot, Enkira, or any company** — this is academic
- Conventional commits: `feat:`, `fix:`, `data:`, `docs:`, `exp:`
- Commit after each meaningful change

## Key Files
| File | Purpose |
|------|---------|
| progress.txt | Session progress log |
| data/synthetic/ | Generated audio datasets |
| src/data_generation/ | TTS data generation scripts |
| k8s/ | OpenShift deployment manifests |
| .env | Cluster login token (gitignored) |
| ACKNOWLEDGMENTS.md | Funding & affiliation credits |
