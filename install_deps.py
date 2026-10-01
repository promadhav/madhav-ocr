"""Keep the exact base-image Torch stack while resolving other dependencies."""
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

before = {name: version(name) for name in ("torch", "torchvision")}
assert "+rocm" in before["torch"], before
constraints = Path("/app/rocm-constraints.txt")
constraints.write_text("".join(f"{name}=={value}\n" for name, value in before.items()))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "--constraint", str(constraints), "-r", "/app/requirements.txt"], check=True)
after = {name: version(name) for name in before}
assert after == before, (before, after)
subprocess.run([sys.executable, "-c",
    "import torch, torchvision; from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; "
    "assert torch.version.hip; print('ROCm imports verified:', torch.__version__)"], check=True)
