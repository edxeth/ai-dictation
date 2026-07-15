# Spec: upgrade PyTorch stack for Local AI Dictation

## Goal
Upgrade the project from the current PyTorch CUDA stack to the latest practical GPU-supported stack, keep NeMo/Parakeet working, keep Whisper working, and fully validate the app after the dependency change.

This spec is intended to be pasted into a new chat as the implementation brief.

---

## Compatibility assessment

### Current installed stack
- Python: 3.12
- `nemo-toolkit`: **2.7.2**
- `torch`: **2.9.1+cu128**
- `torchaudio`: **2.9.1+cu128**
- `torchvision`: **0.24.1+cu128**
- `lightning`: **2.4.0**
- `pytorch-lightning`: **2.6.1**
- GPU: **RTX 3060 Laptop GPU**
- Compute capability: **8.6**
- NVIDIA driver: **595.58.03**
- `nvidia-smi` CUDA version: **13.2**

### Latest versions found
- `torch`: **2.11.0**
- `torchaudio`: **2.11.0**
- `torchvision`: **0.26.0**

### Official dependency constraints found
- `nemo-toolkit 2.7.2` declares: **`torch>=2.6.0`**
- `nemo-toolkit 2.7.2` declares: **`lightning<=2.4.0,>2.2.1`** for ASR-related extras
- NeMo upstream README says: **Python 3.12+** and **PyTorch 2.6+**
- `lightning 2.4.0` allows: **`torch<4.0,>=2.1.0`**

### Practical conclusion
**Yes, upgrading to the latest PyTorch is solver-compatible with the current NeMo version.**

Nothing in the published package metadata blocks:
- `torch 2.11.0`
- `torchaudio 2.11.0`
- `torchvision 0.26.0`
- current GPU / driver

### Risk level
**Moderate, not high.**

Reason:
- the dependency metadata allows it
- GPU support is fine
- but NeMo runtime compatibility with the newest torch should still be treated as **"validate in this repo"**, not assumed
- especially around Parakeet model load / transcribe behavior and CUDA memory behavior

---

## Recommendation

### Recommended target stack
Move to:
- `torch==2.11.0+cu129`
- `torchaudio==2.11.0+cu129`
- `torchvision==0.26.0+cu129`

Reason:
- this is the latest GPU wheel line available from the official PyTorch CUDA indexes
- your driver is new enough
- your Ampere GPU is supported

### Additional recommendation
Pin NeMo explicitly for stability:
- `nemo_toolkit[asr]==2.7.2`

Reason:
- the repo currently works with this version
- avoid future silent resolver drift from unpinned `nemo_toolkit`

### Optional but recommended cleanup
Consider making the environment deterministic by pinning:
- `lightning==2.4.0`

Only do this if needed after dependency resolution; NeMo already constrains it, but explicit pinning can make the environment less surprising.

---

## Required implementation work

### 1. Update dependency declarations
Update at least:
- `pyproject.toml`
- `requirements.txt`

Goal:
- pin `nemo_toolkit[asr]==2.7.2`
- note that PyTorch itself is still installed separately from the official PyTorch CUDA wheel index

Do **not** rely on plain PyPI `torch` if the implementation is meant to preserve GPU acceleration. Use the PyTorch CUDA wheel index.

---

### 2. Reinstall the GPU stack cleanly
Use a clean reinstall for the PyTorch trio.

Recommended commands:

```bash
uv pip uninstall --python .venv/bin/python torch torchaudio torchvision
uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -e .
```

If the implementation sees wheel-resolution issues with `cu129`, fallback plan:

```bash
uv pip install --python .venv/bin/python torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

But prefer trying `cu129` first.

---

### 3. Verify Python-side version state after install
Add a quick verification step that prints:
- torch version
- torch CUDA version
- cuDNN version
- CUDA availability
- GPU name
- NeMo version

Example:

```bash
.venv/bin/python - <<'PY'
import torch, nemo, lightning
print('torch', torch.__version__)
print('torch cuda', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
print('nemo', getattr(nemo, '__version__', 'unknown'))
print('lightning', lightning.__version__)
PY
```

---

### 4. Run the full repo test suite
Required:

```bash
.venv/bin/python -m pytest -q
```

Success criteria:
- all tests pass
- no new import/runtime regressions

---

### 5. Run focused runtime smoke tests
After tests, verify the real app paths.

#### Whisper backend
```bash
local-ai-dictation backend set whisper --restart-bridge
curl -fsS http://127.0.0.1:8765/health
```

Check:
- `model_backend` becomes `whisper`
- bridge loads successfully
- GUI still launches
- Waybar script still reports the right state

#### Parakeet backend
```bash
local-ai-dictation backend set parakeet --restart-bridge
curl -fsS http://127.0.0.1:8765/health
```

Check:
- `model_backend` becomes `parakeet`
- bridge warmup succeeds
- no NeMo import / transcribe crash

Restore default afterward:

```bash
local-ai-dictation backend set whisper --restart-bridge
```

---

### 6. Validate GPU memory behavior
This is important because the whole point is to ensure the new stack does not regress VRAM behavior.

For both backends, record:
- after bridge startup / warmup
- after first transcription
- after backend switch

Use:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

Specific things to observe:
- Whisper should remain the lighter backend
- Parakeet should still load and transcribe successfully
- switching backend should release the old process’s VRAM because the bridge restarts
- upgrading torch must not worsen Parakeet’s initialization spike materially

---

## Code changes to expect
Ideally: **minimal or none**.

But if something breaks after the torch upgrade, check these areas first:
- `src/local_ai_dictation/dictation.py`
- `src/local_ai_dictation/model.py`
- `src/local_ai_dictation/bridge.py`
- NeMo model load path
- any `torch.load` / checkpoint restore behavior

Important note from NeMo docs:
- since PyTorch 2.6, `torch.load` defaults to `weights_only=True`
- current repo already runs on torch 2.9.1, so this is not new, but if a regression appears around checkpoint loading, investigate there first

---

## Documentation updates required
After implementation, update:
- `README.md`

It should reflect:
- latest recommended torch install command
- exact CUDA wheel index used (`cu129` if successful)
- any pinning added for NeMo / Lightning
- whether fallback to `cu128` is still supported/recommended

---

## Success criteria
The upgrade is considered successful only if all of the following are true:

1. `torch`, `torchaudio`, and `torchvision` are upgraded to the target versions
2. `nemo_toolkit[asr]` still imports and Parakeet bridge startup works
3. `local-ai-dictation bridge` works for both Whisper and Parakeet
4. backend switching still restarts the bridge and frees old VRAM
5. GUI model switching still works
6. Waybar left/right click behavior still works
7. full pytest suite passes
8. README is updated to the final dependency/install reality

---

## Fallback plan
If `torch 2.11.0+cu129` causes real runtime regressions in NeMo/Parakeet that are not trivial to fix in this chat:
- fall back to the current working `2.9.1+cu128` stack
- keep the repo otherwise unchanged
- document the attempted upgrade and the exact failing point

---

## Recommended implementation order
1. Pin / update dependency files
2. Reinstall torch stack from official CUDA index
3. Reinstall project editable package
4. Run version sanity checks
5. Run pytest
6. Smoke test Whisper bridge
7. Smoke test Parakeet bridge
8. Validate VRAM behavior
9. Update README

---

## Short decision summary
- **GPU compatibility:** yes
- **NeMo package compatibility on paper:** yes
- **Safe to implement directly:** yes, with validation
- **Recommended target:** `torch 2.11.0 + cu129`
- **Main risk:** runtime behavior in NeMo/Parakeet, not GPU support
