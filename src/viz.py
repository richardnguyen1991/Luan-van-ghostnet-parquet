from __future__ import annotations

import csv
import gc
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize


def _save(figure: plt.Figure, frame: pd.DataFrame, output: Path, name: str) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    frame.round(6).to_csv(output / f"{name}.csv", index=False)
    plt.close(figure)
    gc.collect()
    return [f"{name}.png", f"{name}.pdf", f"{name}.csv"]


def learning_curves(root: Path, out: Path) -> list[str]:
    history = pd.DataFrame(json.loads((root / "history.json").read_text(encoding="utf-8")))
    rows = []
    for item in json.loads((root / "history.json").read_text(encoding="utf-8")):
        for split in ("train", "validation"):
            rows.append({"epoch": item["epoch"], "split": split, **item[split]})
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ("loss", "accuracy", "macro_f1")):
        for split, marker in (("train", "o"), ("validation", "s")):
            subset = frame[frame.split == split]
            axis.plot(subset.epoch, subset[metric], marker=marker, label=split)
        axis.set(xlabel="Epoch", ylabel=metric, title=metric.replace("_", " ").title())
        axis.grid(alpha=.25); axis.legend()
    return _save(fig, frame, out, "learning_curves")


def lr_schedule(root: Path, out: Path) -> list[str]:
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame({"epoch": [x["epoch"] for x in history], "learning_rate": [x["learning_rate"] for x in history]})
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(frame.epoch, frame.learning_rate, marker="o")
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Learning-rate schedule"); ax.grid(alpha=.25)
    return _save(fig, frame, out, "lr_schedule")


def confusion_matrices(root: Path, out: Path) -> list[str]:
    raw = pd.read_csv(root / "confusion_matrix.csv", index_col=0)
    normalized = raw.div(raw.sum(axis=1).replace(0, 1), axis=0)
    files = []
    for name, data, norm in (("confusion_matrix_raw", raw, None), ("confusion_matrix_normalized", normalized, None)):
        fig, ax = plt.subplots(figsize=(max(7, len(raw)*.45), max(6, len(raw)*.4)))
        image = ax.imshow(data.to_numpy(), cmap="Blues", norm=norm); fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=90)
        ax.set_yticks(range(len(data.index)), data.index); ax.set(xlabel="Predicted", ylabel="Actual", title=name.replace("_", " ").title())
        files += _save(fig, data.reset_index(), out, name)
    return files


def probability_curves(root: Path, out: Path) -> list[str]:
    y_true = np.load(root / "y_true.npy"); y_prob = np.load(root / "y_prob.npy")
    mapping = json.loads((root / "label_mapping.json").read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    binary = label_binarize(y_true, classes=np.arange(len(names)))
    if binary.shape[1] == 1:
        binary = np.column_stack((1 - binary[:, 0], binary[:, 0]))
    roc_rows, pr_rows = [], []
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6)); fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for index, name in enumerate(names):
        if index >= binary.shape[1] or binary[:, index].min() == binary[:, index].max():
            continue
        fpr, tpr, _ = roc_curve(binary[:, index], y_prob[:, index])
        precision, recall, _ = precision_recall_curve(binary[:, index], y_prob[:, index])
        roc_rows += [[name, a, b] for a, b in zip(fpr, tpr)]
        pr_rows += [[name, a, b, average_precision_score(binary[:, index], y_prob[:, index])] for a, b in zip(recall, precision)]
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc(fpr,tpr):.3f})")
        ax_pr.plot(recall, precision, label=f"{name}")
    for ax, title, x, y in ((ax_roc,"One-vs-rest ROC","FPR","TPR"),(ax_pr,"One-vs-rest PR","Recall","Precision")):
        ax.set(title=title, xlabel=x, ylabel=y); ax.grid(alpha=.25)
        if len(names) <= 6: ax.legend()
    return _save(fig_roc, pd.DataFrame(roc_rows, columns=["class","fpr","tpr"]), out, "roc_curves") + _save(fig_pr, pd.DataFrame(pr_rows, columns=["class","recall","precision","average_precision"]), out, "pr_curves")


def per_class_metrics(root: Path, out: Path) -> list[str]:
    raw = pd.read_csv(root / "confusion_matrix.csv", index_col=0).to_numpy()
    labels = list(pd.read_csv(root / "confusion_matrix.csv").columns[1:])
    tp=np.diag(raw); support=raw.sum(1); predicted=raw.sum(0)
    frame=pd.DataFrame({"class":labels,"precision":tp/np.maximum(predicted,1),"recall":tp/np.maximum(support,1),"support":support})
    frame["f1"]=2*frame.precision*frame.recall/np.maximum(frame.precision+frame.recall,1e-12)
    fig, ax=plt.subplots(figsize=(max(8,len(labels)*.55),5)); x=np.arange(len(labels)); w=.25
    for offset,key,marker in ((-w,"precision","o"),(0,"recall","s"),(w,"f1","^")): ax.bar(x+offset,frame[key],w,label=key)
    ax.set_xticks(x,labels,rotation=90); ax.set_ylim(0,1); ax.legend(); ax.set_title("Per-class metrics")
    return _save(fig,frame,out,"per_class_metrics")


def class_distribution(root: Path, out: Path) -> list[str]:
    config=json.loads((root/"run_config.json").read_text(encoding="utf-8")); names=config["class_names"]
    rows=[]
    for split,counts in config["class_distribution"].items(): rows += [[split,name,count] for name,count in zip(names,counts)]
    frame=pd.DataFrame(rows,columns=["split","class","count"]); fig,ax=plt.subplots(figsize=(max(8,len(names)*.55),5))
    for split,marker in (("train","o"),("validation","s"),("test","^")):
        subset=frame[frame.split==split]; ax.plot(subset["class"],subset["count"],marker=marker,label=split)
    ax.set_yscale("log"); ax.tick_params(axis="x",rotation=90); ax.legend(); ax.set_title("Class distribution")
    return _save(fig,frame,out,"class_distribution")


def epoch_time(root: Path, out: Path) -> list[str]:
    history=json.loads((root/"history.json").read_text(encoding="utf-8")); frame=pd.DataFrame({"epoch":[x["epoch"] for x in history],"epoch_seconds":[x.get("epoch_seconds_before_checkpoint") for x in history],"checkpoint_seconds":[x.get("checkpoint_and_upload_seconds") for x in history]})
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(frame.epoch,frame.epoch_seconds,marker="o",label="epoch"); ax.plot(frame.epoch,frame.checkpoint_seconds,marker="s",label="checkpoint/upload"); ax.legend(); ax.grid(alpha=.25); ax.set(title="Epoch and checkpoint time",xlabel="Epoch",ylabel="Seconds")
    return _save(fig,frame,out,"epoch_time")


def simple_csv_bar(source: Path, out: Path, name: str, category: str, value: str) -> list[str]:
    frame=pd.read_csv(source).sort_values(value,ascending=False); shown=frame.head(30)
    fig,ax=plt.subplots(figsize=(9,max(4,len(shown)*.25))); ax.barh(shown[category].astype(str)[::-1],shown[value][::-1]); ax.set_title(name.replace("_"," ").title())
    return _save(fig,frame,out,name)


def attention_plot(source: Path, out: Path, name: str, x: str) -> list[str]:
    frame=pd.read_csv(source); pivot=frame.pivot_table(index="sample_index",columns=x,values="weight",aggfunc="mean",fill_value=0)
    fig,ax=plt.subplots(figsize=(9,max(3,len(pivot)*.45))); image=ax.imshow(pivot.to_numpy(),aspect="auto",cmap="viridis"); fig.colorbar(image,ax=ax); ax.set(title=name.replace("_"," ").title(),xlabel=x,ylabel="sample_index")
    return _save(fig,frame,out,name)


def benchmark_plot(root: Path, out: Path) -> list[str]:
    data=json.loads((root/"benchmark.json").read_text(encoding="utf-8")); keys=["forward_latency_ms_p50","forward_latency_ms_p95","forward_latency_ms_mean"]
    frame=pd.DataFrame({"metric":keys,"value":[data[k] for k in keys]}); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(frame.metric,frame.value); ax.tick_params(axis="x",rotation=20); ax.set(ylabel="Milliseconds",title="CPU deployment benchmark")
    return _save(fig,frame,out,"deployment_benchmark")


def generate_all(root_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root=Path(root_dir); out=Path(output_dir); produced={}; skipped={}
    jobs: dict[str, tuple[list[Path], Callable[[], list[str]]]]={
        "learning_curves":([root/"history.json"],lambda:learning_curves(root,out)),
        "lr_schedule":([root/"history.json"],lambda:lr_schedule(root,out)),
        "confusion_matrices":([root/"confusion_matrix.csv"],lambda:confusion_matrices(root,out)),
        "roc_pr_curves":([root/"y_true.npy",root/"y_prob.npy",root/"label_mapping.json"],lambda:probability_curves(root,out)),
        "per_class_metrics":([root/"confusion_matrix.csv"],lambda:per_class_metrics(root,out)),
        "class_distribution":([root/"run_config.json"],lambda:class_distribution(root,out)),
        "epoch_time":([root/"history.json"],lambda:epoch_time(root,out)),
        "cfaco_convergence":([root/"cfaco_convergence.csv"],lambda:simple_csv_bar(root/"cfaco_convergence.csv",out,"cfaco_convergence","iteration","fitness")),
        "feature_importance":([root/"feature_importance.csv"],lambda:simple_csv_bar(root/"feature_importance.csv",out,"feature_importance","feature","integrated_gradients_mean_abs")),
        "spatial_attention":([root/"spatial_attention.csv"],lambda:attention_plot(root/"spatial_attention.csv",out,"spatial_attention","node_index")),
        "temporal_attention":([root/"temporal_attention.csv"],lambda:attention_plot(root/"temporal_attention.csv",out,"temporal_attention","timestep")),
        "ablation_comparison":([root/"ablation_comparison.csv"],lambda:simple_csv_bar(root/"ablation_comparison.csv",out,"ablation_comparison","variant","macro_f1")),
        "deployment_benchmark":([root/"benchmark.json"],lambda:benchmark_plot(root,out)),
    }
    for name,(required,job) in jobs.items():
        missing=[str(path.name) for path in required if not path.exists()]
        if missing: skipped[name]={"reason":"missing_real_artifact","missing":missing}; continue
        try: produced[name]=job()
        except Exception as exc: warnings.warn(f"Plot {name} failed: {type(exc).__name__}: {exc}"); skipped[name]={"reason":"plot_error","error":f"{type(exc).__name__}: {exc}"}
    status={"produced":produced,"skipped":skipped,"no_fake_artifacts":True}
    (out/"report_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    return status
