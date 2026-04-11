# CosyVoice3 + vLLM-Omni Deployment Research

## Problem
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` fails to load in vLLM-Omni v0.18.0 with:
```
ValueError: Failed to read config.json for model: ... Error: config.json found but missing 'model_type'
```

## Root Cause
The `config.json` on HuggingFace is literally `{}` — an **empty JSON object**. It contains no `model_type`, no `architectures`, nothing. This is not a vLLM-Omni bug — it's a model packaging issue from FunAudioLLM.

Source: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/raw/main/config.json → returns `{}`

## Known Issues
1. **vllm-omni #2043** — "[Bug]: cannot run Cosyvoice3 offline with ValueError: This model does not support generation"
   - https://github.com/vllm-project/vllm-omni/issues/2043
   - User tried modifying config.json but still failed
   - Suggested fix: use latest main branch or wait for v0.18.0 stable

2. **CosyVoice #1689** — Related to model_type not being recognized
   - https://github.com/FunAudioLLM/CosyVoice/issues/1689

3. **CosyVoice #1611** — `cosyvoice.yaml not found` workaround
   - https://github.com/FunAudioLLM/CosyVoice/issues/1611

## Potential Solutions

### Option 1: Patch config.json manually
Create a proper `config.json` with `model_type: "cosyvoice3"` and upload to the model directory before serving. This would require:
- Knowing the exact fields vLLM-Omni expects
- Possibly adding `architectures` field

### Option 2: Use CosyVoice's own serving (not vLLM-Omni)
CosyVoice has its own serving options:
- **FastAPI server**: Built into the CosyVoice repo
- **gRPC server**: Docker-based deployment
- **Python SDK**: Direct inference via `CosyVoice2` class

This avoids the vLLM-Omni integration issue entirely but means we'd need a separate container image.

### Option 3: Wait for upstream fix
The v0.18.0 release was expected end of March 2026. The issue may already be addressed in the latest main branch of vllm-omni.

### Option 4: Use CosyVoice2 instead of CosyVoice3
`FunAudioLLM/CosyVoice2-0.5B` may have a proper config.json. CosyVoice2 also supports multilingual + dialect/accent control.

## CosyVoice Official Deployment Instructions
From https://github.com/FunAudioLLM/CosyVoice:

```python
# Direct Python inference (no vLLM needed)
from cosyvoice.cli.cosyvoice import CosyVoice2
model = CosyVoice2('Fun-CosyVoice3-0.5B-2512', load_jit=False, load_trt=False)

# Zero-shot voice clone
for i, j in enumerate(model.inference_zero_shot(
    'text to synthesize',
    'reference transcript',
    reference_audio
)):
    pass
```

## Recommendation
**Short term**: Skip CosyVoice3 on vLLM-Omni. Use Qwen3-TTS (working) + Fish Speech S2 Pro (working) for now.

**Medium term**: Try deploying CosyVoice via its own FastAPI server in a custom container, bypassing vLLM-Omni entirely.

**Long term**: Monitor vllm-omni releases for proper CosyVoice3 support.
