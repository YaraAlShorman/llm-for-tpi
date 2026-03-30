# Running Qwen3 on UW Hyak (Klone)

Guide for running Qwen3-14B on Hyak using your `stf` account and `gpu-l40s` partition.

## Your GPU Resources

| Partition | VRAM/card | Free GPUs | Good for |
|-----------|-----------|-----------|----------|
| `gpu-l40s` | 48 GB | 4 | **Best for Qwen3-14B** |
| `gpu-l40` | 48 GB | 4 | Also works well |
| `gpu-2080ti` | 11 GB | 6 | Too small for 14B |

## Files

| File | Purpose |
|------|---------|
| `qwen3.def` | Apptainer container recipe |
| `server.py` | FastAPI inference server (OpenAI-compatible) |
| `01_build.slurm` | Build the container image |
| `02_download_model.slurm` | Download model weights to gscratch |
| `03_serve.slurm` | Run the inference server |
| `04_interactive.sh` | Interactive GPU shell for testing |

## Step-by-Step

### 1. Upload files to Hyak

```bash
scp qwen3.def server.py *.slurm *.sh yarasho@klone.hyak.uw.edu:~/qwen3/
```

### 2. Build the container (~15-30 min)

```bash
ssh yarasho@klone.hyak.uw.edu
cd ~/qwen3
sbatch 01_build.slurm
```

Monitor progress with `squeue -u yarasho`. When done, the container will be at `/gscratch/stf/containers/qwen3.sif`.

### 3. Download the model (~30-60 min)

```bash
sbatch 02_download_model.slurm
```

This downloads ~28 GB of model weights to `/gscratch/stf/models/`.

### 4. Start the server

```bash
sbatch 03_serve.slurm
```

Check the log to find which node you landed on:

```bash
cat serve_<jobid>.log
# Look for the line: "Node  : g30XX"
```

### 5. Connect from your laptop

Set up an SSH tunnel through the login node to the compute node:

```bash
ssh -L 8000:g30XX:8000 yarasho@klone.hyak.uw.edu
```

Replace `g30XX` with the actual node name from the log. Then query the API:

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello from Hyak!"}],
    "max_tokens": 256
  }'
```

Or use the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Interactive Testing

If you want to poke around before running the full server:

```bash
bash 04_interactive.sh gpu-l40s 1
```

This drops you into a shell inside the container on a GPU node. You can verify the GPU works:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Troubleshooting

**"Disk quota exceeded" during build** — The build scripts already set `APPTAINER_CACHEDIR=/tmp` to avoid the 10 GB home directory limit. If building interactively, run `export APPTAINER_CACHEDIR=/tmp` first.

**Out of GPU memory** — Reduce `MAX_MODEL_LEN` (e.g., set to 4096), or switch to a quantized model like `Qwen/Qwen3-14B-GPTQ-Int4`.

**Job pending (PD state)** — Try `ckpt-all` instead: `sbatch --partition=ckpt-all 03_serve.slurm`. Checkpoint jobs use idle GPUs across the cluster but can be preempted.

**Can't connect to the server** — Double-check the node name in `squeue -u yarasho` matches your SSH tunnel target. Also verify the server actually started by checking the log file.

**Model download interrupted** — Just re-run `sbatch 02_download_model.slurm`. It uses `resume_download=True` and will pick up where it left off.
