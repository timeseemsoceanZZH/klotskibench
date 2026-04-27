# Klotski-Bench

Klotski-Bench is a **benchmarking toolkit** for **Klotski (Hua Rong Dao / sliding-block) puzzle** tasks, built around a small adapter layer over a **frozen legacy engine** in this repository. It packages generation, evaluation, and reporting so you can run reproducible **mini** and **smoke** workflows and compare model outputs on common JSON shapes.

## What it evaluates

The implementation supports **seven task tracks** (each with its own metrics module under `src/metrics/`):

| Task | Focus (brief) |
|------|----------------|
| **S1** | State validity / invalidity classification |
| **S2** | Error localization (e.g. block and error type) |
| **T1** | Legal action-set prediction |
| **T2** | Next-state prediction after a transition |
| **T3** | Transition validity (label and reason) |
| **R1** | Full trajectory / move sequence toward a reference goal (optimal-depth style cases) |
| **R2** | Trajectory verification against a reference |

The public scripts in this repo focus on **R1**-style **mini** and **smoke** subsets; the full evaluator in `src/evaluate.py` can score all seven when cases and predictions are provided in the expected formats.

## Repository structure

| Path | Role |
|------|------|
| `legacy/` | **Frozen** reference engine: `klotski_state.py`, `klotski_moves.py`, `klotski_bfs.py`. Importable as top-level `klotski_*` (see environment/bootstrap). **Do not** treat as part of the mainline `src/` contract layer. |
| `src/` | Mainline benchmark: adapters, generators, tasks, metrics, evaluation, reporting. |
| `scripts/` | Runnable workflows (subset builders, Ollama-backed runs, smoke helpers, path bootstrap). Run commands **from the repository root**. |
| `tests/` | `pytest` suite for the benchmark modules (uses legacy **stubs** in `conftest.py`, not a full physics check of `legacy/`). |
| `output/` | **Not committed** (see `.gitignore`). Created by scripts: predictions, summaries, and reports. |

## Environment setup

- **Python:** 3.10+ recommended (3.11+ fine on Linux servers).
- **Install (pip):** from the repo root:

  ```bash
  pip install -r requirements.txt
  ```

  The benchmark **runtime** uses only the standard library and in-repo `legacy/`. The only pip dependency listed is **pytest** for the test command.

- **Conda (optional):**

  ```bash
  conda env create -f environment.yml
  conda activate klotskibench
  ```

## Assumptions and external dependencies

- **Legacy engine:** The sliding-block BFS, state, and move logic used at runtime for generators live under `legacy/`. The `src.core` adapters import `klotski_bfs`, `klotski_state`, and `klotski_moves` as separate modules. Scripts add the repo root and `legacy/` to `sys.path` via `scripts/_bootstrap_paths.py`.
- **Optional override:** If you mount a different engine, set `KLOTSKIBENCH_EXTRA_SYS_PATH` to a directory of dist roots (documented in `scripts/_bootstrap_paths.py`). Not required for the default in-repo layout.
- **Ollama (local LLM runs):** Benchmark scripts that call models use the **Ollama HTTP API** (default base URL `http://127.0.0.1:11434`). **Ollama is not a Python package** here; install the [Ollama](https://ollama.com) service on the host and **pull** models yourself (e.g. `ollama pull <name>`). Use **exact** model names as shown by `ollama list` (or `python scripts/mini_bench_run_ollama.py list`); wrong names return HTTP 404 and empty predictions.
- **No GPU in Python path:** This repository does not pin CUDA or torch. **GPUs** help only the **inference** backend you choose (e.g. Ollama with local GPU if configured); the benchmark code itself is CPU-only.

## Smoke test workflow (6-case R1 subset)

From the **repository root**:

```bash
# 1) Build a fixed 6-case subset (depths 1..3, 2 per depth, default seed 42)
python scripts/build_smoke_r1_subset.py

# 2) Run a model (default: phi4-mini:latest; set Ollama and model to match your machine)
python scripts/run_smoke_ollama_r1.py
# Or dry-run (no HTTP; still writes report JSON)
python scripts/run_smoke_ollama_r1.py --dry-run

# 3) List result files and preview eval JSON (pass --model if the filename suffix does not match defaults)
python scripts/inspect_smoke_r1.py
```

Default output locations are under `output/smoke_r1/` (e.g. subset JSON, run directory with `eval_summary_<slug>.json`, `predictions_<slug>.json`, `depth_wise_report_<slug>.json`).

## Mini benchmark workflow (configurable R1 subset)

From the **repository root**:

```bash
# Build a custom JSON subset (defaults: depths 1–10, 10 per depth → many cases; use flags to shrink)
python scripts/mini_bench_build_subset.py --out output/mini_r1/mini_r1_subset.json

# List local Ollama models
python scripts/mini_bench_run_ollama.py list

# Run one model and write the same class of reports as the smoke script
python scripts/mini_bench_run_ollama.py run \
  --subset output/mini_r1/mini_r1_subset.json \
  --model <name-from-ollama-list> \
  --out-dir output/mini_r1/run_<name>
```

## Testing

```bash
pytest
```

(From repo root, with `requirements.txt` installed.)

### Task-pipeline validation (all seven tasks)

A separate script (not part of `pytest` stubs) builds **short** case lists with the real `legacy/` engine, checks **oracle-like** and **intentionally wrong** predictions per task, runs `evaluate_benchmark` on the oracle, and prints a pass/fail table.

```bash
python scripts/validate_task_pipelines.py
```

- **What “PASS” means here:** the builder produced cases, the headline metrics are perfect on oracle-style predictions, strictly worse on the provided wrong predictions, and the unified evaluator does not throw for the oracle batch. This is a **regression / wiring** check, not a proof that every edge case in production is covered, and it does not exercise LLM inference.

- **Interpreting failures:** a `FAIL` cell often points to generator coverage, a metrics edge case, or a real bug—investigate the printed details before changing benchmark contracts.

## Output files (where to look)

After a successful **Ollama run** (not `--dry-run`), the chosen `--out-dir` typically contains, for a sanitized model **slug** derived from the model string:

- `predictions_raw_<slug>.json` — raw HTTP responses per case  
- `predictions_<slug>.json` — parsed trajectories in case order  
- `eval_summary_<slug>.json` — aggregated evaluation  
- `main_task_table_<slug>.json` — task table export  
- `depth_wise_report_<slug>.json` — depth-wise R1 report when applicable  

The smoke and mini runners share this pattern.

## Running on another server (e.g. Ubuntu 22.04, NVIDIA)

- **OS:** Ubuntu 22.04 LTS (or similar) is fine. Install Python 3.10+ and `git clone` this repository.
- **GPU:** **Optional** for the benchmark code. **Required** only if your inference stack (e.g. Ollama) is configured to use a GPU; otherwise CPU inference may be slow but valid for small subsets.
- **Ollama:** Start the Ollama service, expose it on a reachable host/port if not localhost, and pass `--ollama-base` to the run scripts if needed. Pull models on that machine before running non-dry-run jobs.
- **Repository root:** All documented `python scripts/...` commands assume the **current working directory** is the project root (where `src/` and `scripts/` sit).

## Quick verification checklist

After clone and `pip install -r requirements.txt`:

1. [ ] `pytest` passes.  
2. [ ] `python scripts/validate_task_pipelines.py` exits 0 and shows all seven tasks as PASS.  
3. [ ] `python scripts/build_smoke_r1_subset.py` creates `output/smoke_r1/smoke_r1_subset.json`.  
4. [ ] `python scripts/run_smoke_ollama_r1.py --dry-run` writes JSON under `output/smoke_r1/run_phi4-mini/` (or your `--out-dir`).  
5. [ ] `python scripts/inspect_smoke_r1.py` reports result files (use `--model` or `--run-dir` as needed).  
6. [ ] (Optional) Ollama installed, model pulled, and a non-dry-run smoke completes with non-empty `predictions_*.json` for your model.

## License / citation

Add your preferred license and citation if publishing the benchmark publicly.
