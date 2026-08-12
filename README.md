# GC-LSTM-GhostNet on CIC-DDoS2019

This folder is the controlled reproduction project for the detection and
classification phase of the GC-LSTM-GhostNet paper workflow. RL-ARL and the
double-trapdoor extension are explicitly out of scope for v1.

Step 2 currently provides:

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
- schema, split leakage, and preprocessing tests.

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

The bundled private Kaggle notebook retrieves `KAGGLE_API_TOKEN` through
Kaggle Secrets and passes it only to the dataset-download subprocess. The
secret value is never printed or stored in project artifacts.

Run tests:

```bash
pytest -q
```

Graph construction, sequence tensors, CFACO, and model code intentionally begin
in later approved steps. No graph adjacency is fabricated in Step 2.
