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

## GPU Operations — ⚠️ MUST SHUT DOWN AFTER USE ⚠️

GPU nodes cost money when pods are running. **Always scale to 0 when done.**

### Login to Cluster
```bash
source .env && eval "$LOGIN"    # Token expires periodically — ask Bo for new one
oc whoami                        # Verify: should show t-9bwen@uchicago.edu
```

### Start a TTS Model
```bash
# Scale up the deployment you need (replicas 0 → 1)
oc scale deployment qwen3-tts --replicas=1      # Qwen3-TTS CustomVoice
oc scale deployment qwen3-tts-base --replicas=1  # Qwen3-TTS Base (voice clone)
oc scale deployment fish-speech --replicas=1      # Fish Speech S2 Pro
oc scale deployment voxtral-tts --replicas=1      # Voxtral TTS

# Wait for pod to be ready (model loading takes 3-8 min)
oc get pods -w   # Watch until STATUS=Running, READY=1/1

# Check logs for "Application startup complete"
oc logs -l app=qwen3-tts --tail=3
```

### Connect to the Model
```bash
# Port-forward to access the API locally
oc port-forward svc/qwen3-tts 8091:8000 &       # Qwen3 CustomVoice → localhost:8091
oc port-forward svc/qwen3-tts-base 8094:8000 &   # Qwen3 Base → localhost:8094
oc port-forward svc/fish-speech 8093:8000 &       # Fish Speech → localhost:8093
oc port-forward svc/voxtral-tts 8092:8000 &       # Voxtral → localhost:8092

# Verify it's responding
curl -s http://localhost:8091/v1/audio/voices
```

### Generate Audio (Python)
```python
import httpx
r = httpx.post("http://localhost:8091/v1/audio/speech", json={
    "input": "Hello, how are you?",
    "voice": "vivian",
    "response_format": "wav",
    "instructions": "你在跟餐馆打电话，速度快一点，带着chinglish的口音。",  # For accent
}, timeout=120)
with open("output.wav", "wb") as f:
    f.write(r.content)
```

For voice clone (Qwen3 Base or Fish Speech):
```python
import base64
with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()
r = httpx.post("http://localhost:8093/v1/audio/speech", json={
    "input": "Text to synthesize",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": f"data:audio/wav;base64,{ref_b64}",
    "ref_text": "Transcript of the reference audio.",
}, timeout=300)
```

### ⚠️ SHUT DOWN GPU — DO THIS IMMEDIATELY AFTER GENERATION ⚠️
```bash
# Scale ALL deployments back to 0
oc scale deployment qwen3-tts --replicas=0
oc scale deployment qwen3-tts-base --replicas=0
oc scale deployment fish-speech --replicas=0
oc scale deployment voxtral-tts --replicas=0

# Verify no pods running
oc get pods   # Should show "No resources found"
```

**Never leave pods running overnight or between sessions. Each H100 GPU costs ~$3/hr.**

### Available Deployments (k8s/ manifests)
| Deployment | Model | Port | Status |
|---|---|---|---|
| qwen3-tts | Qwen3-TTS-12Hz-1.7B-CustomVoice | 8091 | ✅ Working — best for Mandarin/Japanese accent |
| qwen3-tts-base | Qwen3-TTS-12Hz-1.7B-Base | 8094 | ⚠️ Voice clone only, ref_text alignment issues |
| voxtral-tts | Voxtral-4B-TTS-2603 | 8092 | ❌ Cross-lingual too slow, speed param breaks quality |
| fish-speech | fishaudio/s2-pro | 8093 | ❌ Clone timbre only, no accent; needs manual deps |
| cosyvoice3 | Fun-CosyVoice3-0.5B-2512 | - | ❌ Deploy fails (empty config.json + version mismatch) |

## TTS Model Notes
- **Qwen3-TTS CustomVoice**: Best for Mandarin + Japanese accent. Use Chinese instruction: "你在跟餐馆打电话，速度快一点，带着chinglish的口音。" Good voices: serena, dylan, ono_anna, aiden.
- **Voxtral TTS**: Accent quality good but speaking rate too slow for non-English presets. speed>1.0 creates hollow artifacts.
- **Fish Speech S2 Pro**: Only clones timbre, not accent. Inline tags [with strong accent] have minimal effect.
- Always test with small batch first, get human feedback, then scale.

## GitHub Auth
Before any `gh` command in this repo:
```bash
source .envrc   # Sets GH_TOKEN for bwen-uchicago account
```
This ensures issues/PRs are created by `bwen-uchicago`, not personal accounts.

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
