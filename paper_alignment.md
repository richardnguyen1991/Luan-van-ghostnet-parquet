# Paper alignment draft

| thanh_phan | phat_bieu_trong_bai_bao | quyet_dinh_trien_khai | trang_thai |
|---|---|---|---|
| Dataset | 84 characteristics; balanced and imbalanced variants are reported | Use the observed 88-column CIC-DDoS2019 source schema without forcing 84 columns | paper_inferred |
| Min-max scaling | Normalize features to [0,1] | Fit on train groups only; transform validation/test without refitting | paper_exact |
| Missing values | GAIN | Real train-only GAIN implementation in paper_faithful; train-only median in practical_baseline | paper_exact / operational_assumption |
| Outliers | Isolation Forest | Fit on imputed train only; remove train outliers only | paper_exact / operational_assumption |
| Temporal ordering | Temporal traffic patterns and SAX | Preserve capture/file/row provenance; groups assigned before window creation | operational_assumption |
| GCN graph | Network entities and communication relationships | Step 3 uses window-local nodes from namespaced SHA-256 Source/Destination IP hashes and one directed edge per observed flow; no unobserved topology is invented | operational_assumption |
| Temporal window | LSTM processes temporal packet/flow sequences | Step 3 defaults to 16 consecutive source rows with stride 8, never crossing a source-row gap, contiguous group, or split | operational_assumption |
| ResNet-50 | Pretrained high-level feature extraction | Disabled until a meaningful tabular representation and ablation are approved | not_reproduced |
| Class labels | Paper lists VANET attack categories | Preserve the 19 observed CIC-DDoS2019 labels; never rename them to paper labels | operational_assumption |
| RL-ARL and trapdoor | Detection mitigation and security extension | Excluded from v1 detection/classification pipeline | not_reproduced |
| Training device | The paper does not provide a reproducible device/runtime contract | CPU-only execution per explicit user requirement; CUDA and mixed precision are disabled | operational_assumption |
