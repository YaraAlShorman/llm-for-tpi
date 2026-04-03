# LLM for Test Point Insertion (TPI)

Runs Qwen3-14B on the Tillicum cluster (H200 GPUs) to perform automated Test Point Insertion on synthesised gate-level Verilog netlists.

## Files

| File | Purpose |
|------|---------|
| `qwen3.def` | Apptainer container recipe |
| `server.py` | FastAPI inference server (OpenAI-compatible) |
| `01_build.slurm` | Build the container image |
| `02_download_model.slurm` | Download model weights |
| `03_serve.slurm` | Run the inference server |
| `tpi/tpi_insert.py` | Insert test points into a single design |
| `tpi/batch_tpi.py` | Batch TPI across multiple designs |
| `tpi/evaluate.py` | Measure fault coverage before/after TPI |

## Setup (one-time)

All files live under `/gpfs/projects/rhlab/yarasho/`. Clone the repo there before running anything:

```bash
cd /gpfs/projects/rhlab/yarasho
git clone <repo-url> llm-for-tpi
cd llm-for-tpi
```

### 1. Build the container

```bash
sbatch 01_build.slurm
```

Monitor with `squeue -u yarasho`. When done, the container is at:
`/gpfs/projects/rhlab/yarasho/containers/qwen3.sif`

### 2. Download the model

```bash
sbatch 02_download_model.slurm
```

Downloads Qwen3-14B (~28 GB) to `/gpfs/projects/rhlab/yarasho/models/`.
To download a different variant pass the model ID as an argument:

```bash
sbatch 02_download_model.slurm Qwen/Qwen3-7B
```

### 3. Start the inference server

```bash
sbatch 03_serve.slurm
```

The server runs with **INT4 quantization** by default (~7 GB VRAM), leaving ~134 GB free on the H200 for KV cache — enough to fit full netlists in context without truncation.

Check the log to find your node:

```bash
tail -f serve_<jobid>.log
# Look for: "Node  : <nodename>"
```

#### Configuration options

Override defaults via environment variables:

```bash
# Use bf16 instead of INT4 (higher quality, uses ~28 GB)
QUANTIZATION= sbatch 03_serve.slurm

# Larger context window (default: 32768)
MAX_MODEL_LEN=65536 sbatch 03_serve.slurm

# Run the 7B model instead
MODEL_ID=Qwen/Qwen3-7B sbatch 03_serve.slurm
```

### 4. Connect from your laptop

Set up an SSH tunnel through the login node to the compute node:

```bash
ssh -L 8000:<nodename>:8000 <user>@<tillicum-login-node>
```

Test the connection:

```bash
curl http://localhost:8000/health
```

---

## TPI Usage

The `tpi/` scripts run locally and talk to the server over the SSH tunnel.
Yosys must be installed locally (`brew install yosys` / `apt install yosys`).

### `tpi_insert.py` — single design

Synthesises a Verilog RTL file with Yosys, sends the full gate-level netlist to
Qwen3, and applies the selected test points.

```bash
cd tpi

# Basic usage
python tpi_insert.py design.v

# Specify top-level module
python tpi_insert.py design.v --top mymodule

# Custom output path
python tpi_insert.py design.v -o design_tpi.v

# Input is already gate-level (skip synthesis)
python tpi_insert.py design.v --no-synth

# Control number of test points inserted
python tpi_insert.py design.v --cp 6 --op 6

# Point at a different server
python tpi_insert.py design.v --url http://localhost:8000/v1
```

### `batch_tpi.py` — multiple designs

```bash
python batch_tpi.py benchmarks/*.v --cp 6 --op 6
```

### `evaluate.py` — measure fault coverage

```bash
# Evaluate a single netlist
python evaluate.py netlist.v

# Compare before and after TPI
python evaluate.py design_tpi.v --compare design_synth.v

# More patterns for higher accuracy
python evaluate.py netlist.v -n 1000
```

### Typical workflow

```bash
# 1. Open SSH tunnel to the compute node
ssh -L 8000:<nodename>:8000 <user>@<tillicum-login-node>

# 2. Insert test points
python tpi_insert.py my_design.v --top my_module

# 3. Evaluate improvement
python evaluate.py my_design_tpi.v --compare my_design_synth.v
```

---

## GPU / VRAM reference (H200 — 141 GB)

| Model | Precision | Weights | VRAM free for KV cache |
|-------|-----------|---------|------------------------|
| Qwen3-14B | INT4 | ~7 GB | ~134 GB |
| Qwen3-14B | bf16 | ~28 GB | ~113 GB |
| Qwen3-32B | INT4 | ~18 GB | ~123 GB |
| Qwen3-32B | bf16 | ~64 GB | ~77 GB |

---

## Troubleshooting

**Out of GPU memory** — Unlikely on H200 with INT4, but if it happens reduce `MAX_MODEL_LEN` or switch to a smaller model (`Qwen/Qwen3-7B`).

**Model download interrupted** — Re-run `sbatch 02_download_model.slurm`. HuggingFace Hub resumes partial downloads automatically.

**LLM returns invalid JSON** — The script will print the raw response and exit. Try re-running; setting `temperature=0` in the request can also help.

**Can't connect to server** — Verify the node name in the log matches your SSH tunnel. Check the server is still running with `squeue -u yarasho`.
