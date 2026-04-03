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

## Test Point Insertion (TPI)

The `tpi/` directory contains two Python scripts for automated test point insertion
and fault coverage evaluation. These run on your **local machine** and talk to the
Qwen3 server over the SSH tunnel.

### Setup

```bash
cd tpi
pip install -r requirements.txt
```

Yosys must also be installed locally for synthesis (`brew install yosys` on macOS,
`apt install yosys` on Ubuntu).

### `tpi_insert.py` — Insert test points into a design

Synthesises a Verilog RTL file with Yosys, then asks Qwen3 to insert control
and observation test points to maximise fault coverage and minimise pattern count.

```bash
# Synthesise RTL and insert test points (produces design_synth.v + design_tpi.v)
python tpi_insert.py design.v

# Specify the top-level module name
python tpi_insert.py design.v --top mymodule

# Custom output path
python tpi_insert.py design.v -o design_with_tpi.v

# Skip synthesis (input is already a gate-level netlist)
python tpi_insert.py design.v --no-synth

# Use a different model server URL
python tpi_insert.py design.v --url http://localhost:8000/v1
```

### `evaluate.py` — Measure fault coverage and pattern count

Runs stuck-at fault simulation on a gate-level netlist and reports fault coverage
and pattern count. Use `--compare` to see the improvement after TPI.

```bash
# Evaluate a single netlist (500 random patterns by default)
python evaluate.py netlist.v

# Use more patterns for higher accuracy
python evaluate.py netlist.v -n 1000

# Compare TPI netlist against the original (shows coverage delta)
python evaluate.py design_tpi.v --compare design_synth.v

# Load test vectors from a file (one pattern per line, binary values)
python evaluate.py netlist.v --vectors patterns.txt

# Set random seed for reproducibility
python evaluate.py netlist.v --seed 0
```

### Typical workflow

```bash
# 1. Make sure the SSH tunnel to Hyak is open
ssh -L 8000:<node>:8000 yarasho@klone.hyak.uw.edu

# 2. Insert test points
python tpi_insert.py my_design.v --top my_module

# 3. Compare fault coverage before and after
python evaluate.py my_design_tpi.v --compare my_design_synth.v
```

---

## Running on Jetstream2 (g3.xl / A100)

The `jetstream/` directory contains scripts for running the same server on a
Jetstream2 GPU instance (Ubuntu 22.04, A100 40 GB) without SLURM or Apptainer.

### A100 model fit

| Model | Precision | VRAM | Fits on 40 GB A100? |
|-------|-----------|------|----------------------|
| Qwen3-14B | bf16 | ~28 GB | Yes |
| Qwen3-32B | bf16 | ~64 GB | No |
| Qwen3-32B | Int4 | ~18 GB | Yes (add `bitsandbytes`) |

### Step-by-step

**1. Clone the repo on your Jetstream instance**

```bash
ssh ubuntu@<your-instance-ip>
git clone https://github.com/YaraAlShorman/llm-for-tpi.git
cd llm-for-tpi
```

**2. One-time setup** (installs PyTorch, transformers, etc.)

```bash
bash jetstream/00_setup.sh
```

**3. Download the model**

```bash
bash jetstream/01_download_model.sh
# or for a different model:
MODEL_ID=Qwen/Qwen3-32B bash jetstream/01_download_model.sh
```

**4. Start the server**

```bash
bash jetstream/02_serve.sh
# Custom port or context length:
PORT=8080 MAX_MODEL_LEN=16384 bash jetstream/02_serve.sh
```

**5. Connect from your laptop**

```bash
ssh -L 8000:<your-instance-ip>:8000 ubuntu@<your-instance-ip>
curl http://localhost:8000/health
```

The `tpi/` scripts work identically — just make sure the tunnel is open before running `tpi_insert.py`.

---

## Interactive Testing (Hyak)

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
