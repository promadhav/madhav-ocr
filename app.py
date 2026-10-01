"""Challenge 2 CLI. Model inference lives in worker.py, not this process."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time


def request(payload, timeout=28):
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(max(0.01, deadline - time.monotonic()))
        client.connect(os.environ.get("OCR_SOCKET", "/tmp/madhav-ocr.sock"))
        client.sendall(json.dumps(payload).encode() + b"\n")
        data = bytearray()
        while not data.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OCR request exceeded client deadline")
            client.settimeout(remaining)
            chunk = client.recv(65536)
            if not chunk:
                raise RuntimeError("Worker disconnected before answering")
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise RuntimeError("Oversized worker response")
    result = json.loads(data)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-image")
    group.add_argument("--health", action="store_true")
    args = parser.parse_args()
    if args.health:
        print(json.dumps(request({"op": "health"}, timeout=3)))
        return
    path = Path(args.input_image).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Input must be a file")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (path.stem + "_output.json")
    # A failed retry must not leave a stale answer for the same image.
    target.unlink(missing_ok=True)
    response = request({"op": "ocr", "path": str(path)})
    if not isinstance(response.get("text"), str):
        raise ValueError("Worker did not return a text string")
    answer = {"text": " ".join(response["text"].split())}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                         delete=False, suffix=".tmp") as handle:
            temporary = handle.name
            json.dump(answer, handle, ensure_ascii=False)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OCR failed: {exc}", file=sys.stderr)
        sys.exit(1)
