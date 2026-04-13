# Speed results (raw + summary)

Place artifacts here, for example:

- `bench_serving` stdout / stderr logs  
- Exported JSON or CSV from your logging wrapper  
- A short `SUMMARY.md` per hardware / git SHA

## Summary table (template)

| Model | KV config | Output tok/s | TTFT (ms) | TPOT (ms) | ITL (ms) | Workload | Date | Git commit |
|-------|-----------|--------------|-----------|-----------|----------|----------|------|------------|
| — | BF16 / auto | — | — | — | — | random 256→32, conc=16 | — | — |
| — | INT4 | — | — | — | — | same | — | — |
| — | INT4 + BDR K-only | — | — | — | — | same | — | — |

Replace columns with the metrics your `bench_serving` build prints; keep workload columns identical across rows.
