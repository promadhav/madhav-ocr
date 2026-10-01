"""Long-running single-GPU OCR worker. No public network port."""
import json
import os
from pathlib import Path
import socketserver
import time
import traceback

os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("HF_HOME", "/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

INSTRUCTION = (
    "Transcribe the visible text accurately. For a license plate, return ONLY the "
    "registration number. Exclude surrounding state names and slogans, but preserve "
    "Chinese province characters and letters within the registration. For road signs, "
    "return ALL printed text in reading order. Join multiple lines with spaces. "
    "Do not add labels, explanations, or units not printed in the image."
)


class OCR:
    def __init__(self):
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A working AMD ROCm GPU is required; no CPU fallback")
        source = os.environ.get("MODEL_SOURCE", "Qwen/Qwen2.5-VL-3B-Instruct")
        options = {"cache_dir": os.environ["HF_HUB_CACHE"]}
        if not Path(source).is_dir():
            options["revision"] = os.environ.get("MODEL_REVISION", "6628554")
        self.processor = AutoProcessor.from_pretrained(
            source, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, **options
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, torch_dtype=torch.float16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", **options
        ).eval()
        # Finish first-use GPU initialization before accepting requests.
        self.read(Image.new("RGB", (320, 160), "white"), max_tokens=1)

    def read(self, photo, max_tokens=64):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": INSTRUCTION}
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[photo], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        text = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return " ".join(text.split())


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        try:
            line = self.rfile.readline(65537)
            if len(line) > 65536:
                raise ValueError("Request too large")
            message = json.loads(line)
            if message.get("op") == "health":
                result = {"ready": True, "gpu": torch.cuda.get_device_name(0)}
            elif message.get("op") == "ocr":
                start = time.monotonic()
                with Image.open(message["path"]) as image:
                    if image.format not in {"PNG", "JPEG", "TIFF"}:
                        raise ValueError("Supported image formats: PNG, JPEG, TIFF")
                    photo = ImageOps.exif_transpose(image).convert("RGB")
                result = {"text": self.server.ocr.read(photo)}
                print(json.dumps({"inference_seconds": round(time.monotonic() - start, 3)}), flush=True)
            else:
                raise ValueError("Unknown operation")
        except Exception as exc:
            traceback.print_exc()
            result = {"error": str(exc)}
        try:
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    start = time.monotonic()
    socket_path = Path(os.environ.get("OCR_SOCKET", "/tmp/madhav-ocr.sock"))
    # A live worker should never be replaced by another one.
    if socket_path.exists():
        raise RuntimeError(f"Socket already exists: {socket_path}; stop/check the old worker first")
    ocr = OCR()
    with socketserver.UnixStreamServer(str(socket_path), Handler) as server:
        os.chmod(socket_path, 0o600)
        server.ocr = ocr
        print(json.dumps({"ready": True, "startup_seconds": round(time.monotonic() - start, 2),
                          "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}), flush=True)
        try:
            server.serve_forever()
        finally:
            socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
