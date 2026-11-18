"""Comparison CSV + figures from results/models/*_metrics.json (real runs only).
Run: python -m models.make_plots
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
mdir, fdir = out / "models", out / "figures"
fdir.mkdir(parents=True, exist_ok=True)

rows = [json.load(open(f)) for f in sorted(glob.glob(str(mdir / "*_metrics.json")))]
df = pd.DataFrame(rows).drop(columns=["cv_fold_top1", "test_confusion"], errors="ignore")
df.to_csv(out / "model_comparison.csv", index=False)
names = df["model"].tolist()
x = np.arange(len(names))


def bars(cols, labels, title, fname, errs=None):
    plt.figure(figsize=(8, 4))
    w = 0.8 / len(cols)
    for i, (c, l) in enumerate(zip(cols, labels)):
        e = df[errs[i]] if errs and errs[i] else None
        plt.bar(x + i * w, df[c], w, yerr=e, label=l, capsize=3)
    plt.xticks(x + w * (len(cols) - 1) / 2, names, rotation=20)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / fname, dpi=150)
    plt.close()


bars(["cv_macro_f1", "cv_top1"], ["CV macro F1 (thr 0.5)", "CV Top-1"],
     "Grouped 5-fold CV (train trips)", "cv_comparison.png",
     errs=["cv_macro_f1_std", "cv_top1_std"])
bars(["test_top1", "test_macro_f1_pick"], ["Test Top-1", "Test macro F1 (top-1 pick)"],
     "Student-test comparison", "test_top1_f1.png")
ll = df.dropna(subset=["test_log_loss"])
if len(ll):
    plt.figure(figsize=(6, 4))
    plt.bar(ll["model"], ll["test_log_loss"])
    plt.title("Student-test log loss (probability models only)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fdir / "test_log_loss.png", dpi=150)
    plt.close()

# confusion matrices
fig, ax = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.2), squeeze=False)
for a_, r in zip(ax[0], rows):
    cm = np.array(r["test_confusion"])
    a_.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a_.text(j, i, int(v), ha="center", va="center", fontsize=8)
    a_.set_title(r["model"])
    a_.set_xticks([0, 1]); a_.set_yticks([0, 1])
    a_.set_xlabel("pred" if r["model"] != "lgbm_rank" else "pred (top-1 pick)")
plt.tight_layout()
plt.savefig(fdir / "confusion_matrices.png", dpi=150)
plt.close()

# real epoch curves (MLP only)
h = mdir / "mlp_history.json"
if h.exists():
    hist = pd.DataFrame(json.load(open(h)))
    plt.figure(figsize=(6, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train loss (incl. L2)")
    plt.plot(hist["epoch"], hist["val_loss"], label="val log loss (held-out trips)")
    plt.xlabel("epoch"); plt.legend(); plt.title("MLP loss curves")
    plt.tight_layout()
    plt.savefig(fdir / "mlp_loss_curves.png", dpi=150)
    plt.close()
print(df[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1_pick",
          "route_continuity", "test_log_loss", "infer_ms_per_1k_rows"]].to_string())
