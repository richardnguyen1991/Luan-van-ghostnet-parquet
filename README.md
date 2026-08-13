# GC-LSTM-GhostNet on CIC-DDoS2019

This folder is the controlled reproduction project for the detection and
classification phase of the GC-LSTM-GhostNet paper workflow. RL-ARL and the
double-trapdoor extension are explicitly out of scope for v1.

Steps 2 through 4 currently provide:

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
- a directed three-layer GCN, LSTM, temporal attention, Ghost module, and classifier;
- class-balanced AdamW training with cosine scheduling and gradient clipping;
- immutable per-epoch, latest, and best checkpoints with complete restart state;
- accuracy, precision, recall, macro/weighted F1, AUC-ROC, confusion matrix,
  runtime, memory, latency, throughput, and learning-curve artifacts;
- optional secure S3 uploads using environment credentials (never serialized).

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

Step 4 sampled training smoke test (two epochs):

```bash
python train.py \
  --data-dir /kaggle/input/datasets/dungnguyen28101991/cicddos2019-parquet \
  --output-dir /kaggle/working/gc_lstm_ghostnet_step4 \
  --samples-per-file 2048 \
  --epochs 2 \
  --batch-size 64 \
  --device cpu
```

This project is CPU-only by design. The CLI accepts only `--device cpu`, the
training loop never selects CUDA or mixed precision, and the bundled Kaggle
notebook is saved with GPU disabled.

The CLI default is 100 epochs. Add `--upload-checkpoints-to-s3 --s3-bucket
BUCKET --s3-prefix PREFIX --aws-region REGION` to upload checkpoints and final
artifacts. AWS credentials must be supplied through environment variables or
Kaggle Secrets; they are not accepted as command-line arguments.

Step 5 deterministic resume acceptance (CPU only):

```bash
python -m src.step5_resume_smoke \
  --data-dir /kaggle/input \
  --output-dir /kaggle/working/gc_lstm_ghostnet_step5 \
  --samples-per-file 2048 \
  --batch-size 64 \
  --device cpu
```

This executes a three-epoch uninterrupted reference and a second run that
stops after epoch 2 and resumes at epoch 3. Acceptance requires an exact match
of every final model tensor and a contiguous `[1, 2, 3]` history. Checkpoints
include optimizer/scheduler/RNG state plus config, preprocessing, selected
feature, and graph hashes. S3 writes use a temporary object, verify size and
SHA-256 metadata, copy to the final key, then delete the temporary object.

Step 6 artifact/report smoke run:

```bash
python -m src.step6_artifact_smoke \
  --data-dir /kaggle/input \
  --output-dir /kaggle/working/gc_lstm_ghostnet_step6 \
  --device cpu
```

Regenerate figures without training:

```bash
python -m src.make_report \
  --artifact-dir /path/to/artifacts \
  --output-dir /path/to/regenerated-report
```

Every produced figure has PNG (300 dpi), PDF, and source CSV. The smoke run
produces 11 real visualization groups. CFACO convergence and ablation comparison
are explicitly recorded as skipped until those experiments have real artifacts;
the report generator never fabricates placeholder measurements.

The bundled private Kaggle notebook uses only the dataset attached under
`/kaggle/input`. It never requests `KAGGLE_API_TOKEN`. If the input is absent,
the setup cell stops with an explicit **Add Input** instruction. GitHub Actions
secrets are intentionally unavailable inside Kaggle notebooks.

Run tests:

```bash
pytest -q
```

CFACO remains a later step. Step 3 creates a
traceable graph from observed source/destination endpoints; it does not infer
unobserved vehicle topology. Full mixed-group streaming tensor production is
paired with the Step 4 training loop so every eligible train window is consumed
without first materializing the full dataset in memory. The current Step 4
implementation is explicitly labeled `bounded_contiguous_sample`; it refuses
`--full-dataset`/`--stream-files` so a sampled result cannot be mislabeled as a
full-dataset experiment.
