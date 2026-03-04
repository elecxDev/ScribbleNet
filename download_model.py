"""
Run this ONCE before starting the app to download the model locally:
    python download_model.py
"""
import httpx
from huggingface_hub.utils._http import set_client_factory
import huggingface_hub.utils._http as _hf_http

# Disable SSL verification — needed when behind a corporate proxy / VPN
# that performs SSL inspection with a self-signed certificate.
set_client_factory(lambda: httpx.Client(verify=False))
_hf_http._GLOBAL_CLIENT = None  # force fresh client with new factory

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID  = "microsoft/trocr-base-handwritten"
LOCAL_DIR = Path(__file__).parent / "models" / "trocr-base-handwritten"

# Check for the actual weight file, not just the directory
weights_file = LOCAL_DIR / "pytorch_model.bin"
safetensors_file = LOCAL_DIR / "model.safetensors"

if weights_file.exists() or safetensors_file.exists():
    print(f"Model already present at: {LOCAL_DIR}")
else:
    print(f"Downloading {MODEL_ID} → {LOCAL_DIR}")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(LOCAL_DIR))
    print("Done.")

