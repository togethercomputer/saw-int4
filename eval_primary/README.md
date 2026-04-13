# Primary evaluation — accuracy (BF16, INT4, BDR)

This folder holds **accuracy** logs and summary tables for the **primary** paper track: **BF16**, **INT4**, and **BDR** on **`third_party/sglang-fast-rotation`** only.

**Throughput** for the same track lives under [../eval_speed/README.md](../eval_speed/README.md).

**Canonical doc:** [../docs/03-evaluation-matrix.md](../docs/03-evaluation-matrix.md#primary-evaluation-track)  
**Script:** [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh)  
**Client:** [openai/simple-evals](https://github.com/openai/simple-evals)

## Workflow

1. `cd third_party/sglang-fast-rotation/python` and install per [../docs/01-preparation.md](../docs/01-preparation.md).
2. From repo root: `./scripts/run_primary_eval_matrix.sh bf16` (or `int4`, `bdr`, `bdr_kv`) — start the printed server, then run simple-evals with `OPENAI_BASE_URL` pointing at `/v1`.
3. Store outputs under [results/](results/) and mirror headline scores in the main [README.md](../README.md).
