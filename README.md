# GC-LSTM-GhostNet on CIC-DDoS2019

This folder is the controlled reproduction project for the detection and
classification phase of the GC-LSTM-GhostNet paper workflow. RL-ARL and the
double-trapdoor extension are explicitly out of scope for v1.

Steps 2 and 3 currently provide:

- centralized YAML configuration for `paper_faithful` and `practical_baseline`;
- structure-preserving Parquet discovery and manifest validation;
- whitespace-safe schema canonicalization;
- distributed sampled reads across Parquet row groups;
- stable `sample_id` derived from source file and source row;
- contiguous group creation before train/validation/test assignment;
- separate outer-train scenarios 70% and 80%;
- train-only median or GAIN imputation;
- train-only Isolation Forest filtering;
- train-only MinMax scaling and preprocessing artifacts;
- schema, split leakage, and preprocessing tests;
- deterministic contiguous Parquet slices for sequence validation;
- strict windows that never cross source-row gaps, groups, or splits;
- sequence-to-one labels taken from the last flow in each window;
- directed Source IP to Destination IP graph edges for every flow timestep;
- window-local graph node indices backed by namespaced SHA-256 endpoint hashes;
- compressed sequence and graph tensor artifacts with leakage reports.

Default Kaggle dataset:

```text
dungnguyen28101991/cicddos2019-parquet
```

Example sampled audit on Kaggle:

```bash
python -m src.data \
  --data-dir /kaggle/input \
  --output-dir /kaggle/working/Luan-Van-GC-LSTM-GhostNet-CICDDoS2019-v1/outputs/audit \
  --config configs/base.yaml \
  --mode-config configs/practical_baseline.yaml \
  --samples-per-file 2048
```

Complete Step 2 smoke test for both 70% and 80% outer-train scenarios:

```bash
python -m src.step2_smoke \
  --data-dir /kaggle/input \
  --config configs/base.yaml \
  --mode-config configs/practical_baseline.yaml \
  --samples-per-file 2048
```

Complete Step 3 graph/sequence smoke test:

```bash
python -m src.step3_smoke \
  --data-dir /kaggle/input \
  --config configs/base.yaml \
  --mode-config configs/practical_baseline.yaml \
  --samples-per-file 2048 \
  --sequence-length 16 \
  --sequence-stride 8
```

Each split writes `sequence_tensors.npz`, `graph_tensors.npz`, and
`sequence_graph_manifest.json`. The run also writes
`sequence_leakage_report.json` and `step3_summary.json`. Raw IP addresses are
never stored in these tensor artifacts.

The paper does not publish its graph-construction rule, sequence length, or
stride. Step 3 therefore records the configurable 16/8 endpoint-flow design as
an operational assumption rather than presenting it as paper-exact.

The bundled private Kaggle notebook first uses the attached dataset under
`/kaggle/input`. Only when that mount is absent does it retrieve
`KAGGLE_API_TOKEN` through Kaggle Secrets and pass it to the download
subprocess. The secret value is never printed or stored in project artifacts.

Run tests:

```bash
pytest -q
```

CFACO and model training intentionally begin in later steps. Step 3 creates a
traceable graph from observed source/destination endpoints; it does not infer
unobserved vehicle topology. Full mixed-group streaming tensor production is
paired with the Step 4 training loop so every eligible train window is consumed
without first materializing the full dataset in memory.

