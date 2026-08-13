from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark_model
from .config import load_config
from .data import audit_contiguous_sequence_dataset, write_json
from .explainability import generate_explainability_artifacts
from .graph_sequences import SPLIT_NAMES, align_preprocessed_split, build_sequence_graph_tensors, validate_sequence_leakage
from .model import GCLSTMGhostNet
from .preprocessing import LeakageSafePreprocessor
from .splits import assign_group_splits
from .training import GraphSequenceDataset, collate_graph_sequences, train_model
from .viz import generate_all


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Step 6 CPU artifact/report smoke run")
    parser.add_argument("--data-dir",default="/kaggle/input")
    parser.add_argument("--output-dir",default="/kaggle/working/gc_lstm_ghostnet_step6")
    parser.add_argument("--samples-per-file",type=int,default=2048)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--sequence-length",type=int,default=16)
    parser.add_argument("--sequence-stride",type=int,default=8)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=["cpu"],default="cpu")
    parser.add_argument("--benchmark-warmups",type=int,default=50)
    parser.add_argument("--benchmark-measurements",type=int,default=500)
    return parser.parse_args()


def main() -> None:
    args=parse_args(); output=Path(args.output_dir); artifacts=output/"artifacts"; figures=output/"report"
    config=load_config("configs/base.yaml","configs/practical_baseline.yaml")
    config["data"]["samples_per_file"]=args.samples_per_file
    config["step3"]["sequence_length"]=args.sequence_length
    config["step3"]["sequence_stride"]=args.sequence_stride
    sample,profile,manifest=audit_contiguous_sequence_dataset(args.data_dir,output/"audit",config)
    split=assign_group_splits(sample,profile["label_column"],float(config["step4"]["outer_train_fraction"]),float(config["splits"]["validation_fraction_of_outer_train"]),int(config["data"]["sequence_group_rows"]),args.seed,int(config["splits"]["candidate_attempts"]))
    preprocessor=LeakageSafePreprocessor(config,args.seed); processed=preprocessor.fit_transform_splits(split.frame,profile["label_column"])
    bundles={name:build_sequence_graph_tensors(align_preprocessed_split(split.frame,processed,name),name,config) for name in SPLIT_NAMES}
    leakage=validate_sequence_leakage(bundles)
    if leakage["status"]!="passed": raise RuntimeError(leakage)
    run_args=vars(args)|{"run_name":"step6-artifact-smoke","session_id":"step6-cpu"}
    training=train_model(bundles,processed.metadata,config,artifacts,args.epochs,args.batch_size,float(config["future_training_contract"]["learning_rate"]),args.seed,run_args)
    checkpoint_path=artifacts/f"final_model_epoch_{args.epochs:03d}.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    model=GCLSTMGhostNet(bundles["train"].sequence_x.shape[2],len(processed.metadata["label_mapping"]),config)
    model.load_state_dict(checkpoint["model"])
    generate_explainability_artifacts(model,bundles["test"],processed.metadata,artifacts)
    batch=collate_graph_sequences([GraphSequenceDataset(bundles["test"])[0]])
    benchmark=benchmark_model(model,batch,checkpoint_path,artifacts,args.benchmark_warmups,args.benchmark_measurements)
    report=generate_all(artifacts,figures)
    expected_skips={"cfaco_convergence","ablation_comparison"}
    if set(report["skipped"]) != expected_skips:
        raise RuntimeError(f"Unexpected report skips: {report['skipped']}")
    summary={"status":"passed","device":"cpu","training":training,"benchmark":benchmark,"report":report,"expected_not_yet_run":sorted(expected_skips),"sequence_leakage_status":leakage["status"],"sample_manifest":manifest}
    write_json(output/"step6_summary.json",summary); print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
