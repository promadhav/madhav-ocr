# MADHAV Lens - Challenge 2 OCR Docker starter

This is a packaging candidate, NOT a graded or GPU-container-validated submission.
Built around the user's working Qwen2.5-VL-3B-Instruct notebook prototype.
This package does not implement Challenge 3 RAG.

## What has been checked

- User's AMD notebook: GPU math passed; model loaded using 7.07 GiB allocated VRAM.
- Notebook OCR: synthetic STOP and generated ROAD WORK AHEAD answered correctly.
- User-reported latency: STOP 9.27 s, road sign 0.46 s on a later call.
- Package verification: see VALIDATION.txt. Mock contract tests do NOT test model accuracy.
- Not yet checked: Docker build, complete dependency resolution in the base image,
  GPU worker in this package, hidden/representative image accuracy, cold startup,
  final image uncompressed size, or final container on an AMD GPU host.

## Windows build - read before running

Your supplied registry manifest has 20,492,259,842 bytes (20.49 GB / 19.08 GiB)
of compressed base layers. Download and unpack require different amounts of disk.
70 GB free does not guarantee enough build space. Check Docker Desktop's disk
location and capacity before pulling. This download uses the laptop's internet.
No files in this ZIP are model weights or Docker image layers.

1. Extract the ZIP. Open PowerShell inside the folder containing Dockerfile.
2. Confirm `Get-ChildItem` shows Dockerfile, app.py, worker.py, requirements.txt.
3. On a connection with enough data and disk capacity, run:

```powershell
docker build --platform linux/amd64 --progress=plain -t madhav-ocr:mc2-v1 .
if ($LASTEXITCODE -ne 0) { throw "Build failed; do not submit" }
docker image inspect madhav-ocr:mc2-v1 --format '{{.Size}}'
docker run --rm --entrypoint python3 madhav-ocr:mc2-v1 -c "import torch; from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; print(torch.__version__, torch.version.hip)"
```

The import check deliberately overrides the default worker. It requires no GPU
and does not download model weights. It is NOT an inference test.
Check actual uncompressed image size <= 64,424,509,440 bytes (60 GiB).
Do not squash/flatten the image or change the mandated FROM.

## Model download tradeoff

Default INCLUDE_MODEL=0: weights download on the AMD runtime, under /models.
This is permitted by the supplied rules, but download + load + warmup must fit
the 10-minute cold-start gate. This has not been timed. The cache on your notebook
is NOT automatically transferred into the Docker image or grader environment.

Alternative: add `--build-arg INCLUDE_MODEL=1` to the build command to ship
weights inside the image. This adds approximately 7.52 GB of model download on the
build machine, image disk usage and registry upload. Check final image size.
Do not submit the default build without validating its cold-start budget.

## Test worker/client using the existing AMD notebook environment

Upload this ZIP to AMD Jupyter and extract it to /workspace/madhav-ocr-starter.
Use a Jupyter terminal (not a Python cell) and the same Python environment used
in the successful notebook. Preserve the existing HF cache to avoid re-download.
The notebook model may still occupy ~7 GiB; stop its kernel if memory is tight.

```bash
cd /workspace/madhav-ocr-starter
export HF_HOME=/workspace/madhav-lens/models/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export OUTPUT_DIR=/workspace/madhav-lens/output
export OCR_SOCKET=/tmp/madhav-ocr.sock
python3 worker.py > /workspace/madhav-lens/worker.log 2>&1 &
echo $!
```

Wait for `"ready": true` in worker.log. Then, in the SAME terminal:

```bash
python3 app.py --health
time python3 app.py --input-image /workspace/test.png
cat /workspace/madhav-lens/output/test_output.json
```

The client must finish in under 30 s, including its process launch and JSON write.
Each client has a 28 s socket deadline; exceeding it is a failure, not a valid
empty OCR response. The worker performs inference sequentially. A client timeout
does not cancel GPU work; investigate slow images before continuing.
Stop your own worker with `kill PID` (the number printed above) before leaving.
An existing socket is not silently removed. If the worker was killed abruptly,
confirm no worker is still running before removing /tmp/madhav-ocr.sock.

## Final container GPU validation still needs an AMD Docker host

The current hackathon notebook does not expose Docker. Confirm an organizer-
supported container test route or use an authorized AMD Linux Docker host.
Do not assume Windows Docker Desktop exposes this cloud GPU.
On a suitable AMD Linux host with /dev/kfd and /dev/dri:

```bash
mkdir -p input output
# Put independently labeled test PNG/JPEG/TIFF images in input/ first.
docker run -d --name madhav-ocr-test \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --shm-size=2g \
  -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output" \
  madhav-ocr:mc2-v1
docker logs -f madhav-ocr-test
# After ready, in another terminal:
docker exec madhav-ocr-test python3 /app/app.py --health
time docker exec madhav-ocr-test python3 /app/app.py --input-image /app/input/test.png
```

Host GPU permissions/group setup may vary; follow the host operator's setup.
Measure cold startup with no model cache, peak GPU usage throughout, and repeated
CLI time per image. Keep the container running for all inputs. Check 1-48 GiB
VRAM, 10-minute startup and run limits, 30 s per input, and <=60 GiB image size.
Stop only your own test container after testing: `docker stop madhav-ocr-test`.

## Accuracy checks before submission

Use organizer samples and additional separately labeled images: US plate banner
exclusion, Chinese registration prefixes, multi-line signs, digits-only plaques,
blur, glare, low light, angle, small text, large inputs, PNG/JPEG/TIFF.
Two easy generated images are insufficient validation. 64 output tokens and the
1024 visual-token pixel cap are baseline settings, not proven optimal choices.
Answer normalization: uppercase; remove whitespace and - . middle-dot _.
Expected values must come from manual image labels, never model predictions.

## Upload only after checks pass

Create a PUBLIC Docker Hub repository. Replace YOUR_DOCKER_ID below:

```powershell
docker login
docker tag madhav-ocr:mc2-v1 YOUR_DOCKER_ID/madhav-ocr:mc2-v1
docker push YOUR_DOCKER_ID/madhav-ocr:mc2-v1
```

Submit the image reference privately through the organizer's submission flow.
The supplied rules say the image must be publicly pullable but its reference
should not be advertised or committed in a public repository. Verify an
unauthenticated pull before submission. No passwords, tokens or .env files belong
inside the image; .dockerignore and explicit COPY restrict the build context.

## Sources

- User-uploaded All About the Challenges PDF: exact base, CLI, output, limits.
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2_5_vl
- https://docs.docker.com/reference/cli/docker/buildx/build/
