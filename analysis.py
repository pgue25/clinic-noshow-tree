"""Clinic no-shows: should the clinic overbook this slot?

Fits a classification tree (v1) on 70% of the appointments, compares it to the
"everyone shows" baseline on the held-out 30%, applies the overbook/hold rule to
appointments D and E, and puts a dollar value on each policy.

Run:  .venv/bin/python analysis.py
Every number printed here (and saved to output/) is what goes on the slides.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

SEED = 42
TEST_SIZE = 0.30
MAX_DEPTH = 3
# appointment_hour is left out: no-shows and check-ins have the same average hour (12.1 vs 12.2).
FEATURES = ["prior_no_shows", "sms_reminder", "lead_time_days", "new_patient", "distance_miles", "age"]

# Value per appointment (cost table)
VALUE_TN = 0      # predicted show, came            -> hold, fine
VALUE_FP = -60    # predicted no-show, came         -> overbooked, overtime + refund
VALUE_FN = -120   # predicted show, did not come    -> empty slot
VALUE_TP = 0      # predicted no-show, did not come -> second patient fills slot

ANNUAL_VOLUME = 10_000  # assumed appointments per year (~40/day x 250 clinic days)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ---------------------------------------------------------------- 1. data + split
df = pd.read_csv(HERE / "data" / "clinic_appointments.csv")
y = (df["no_show"] == "no-show").astype(int)  # 1 = no-show
X = df[FEATURES]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

log("=== 1. DATA ===")
log(f"Rows: {len(df)}, columns: {df.shape[1]}, no-shows: {y.sum()} ({y.mean():.1%})")
log(f"Train: {len(X_train)} appts, {y_train.sum()} no-shows ({y_train.mean():.1%})")
log(f"Test:  {len(X_test)} appts, {y_test.sum()} no-shows ({y_test.mean():.1%})")
log(f"Random seed: {SEED}, stratified on no_show")

# ---------------------------------------------------------------- 2. fit v1
v1 = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)
v1.fit(X_train, y_train)

log("\n=== 2. MODEL v1 ===")
log(f"Classification tree, max_depth={MAX_DEPTH}, features={FEATURES}")
log("Tree rules (class 1 = no-show):")
log(export_text(v1, feature_names=FEATURES, show_weights=True))
imp = pd.Series(v1.feature_importances_, index=FEATURES).sort_values(ascending=False)
log("Feature importance:")
log(imp.round(3).to_string())

fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(v1, feature_names=FEATURES, class_names=["show", "no-show"], filled=True,
          rounded=True, impurity=False, proportion=False, fontsize=10, ax=ax)
fig.savefig(OUT / "tree_v1.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. evaluate
def metrics(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

base_pred = np.zeros(len(y_test), dtype=int)  # everyone shows
v1_pred = v1.predict(X_test)
results = pd.DataFrame([metrics("baseline", y_test, base_pred), metrics("v1", y_test, v1_pred)])

train_acc = accuracy_score(y_train, v1.predict(X_train))
test_acc = accuracy_score(y_test, v1_pred)

log("\n=== 3. TEST-SET FIT (n=%d) ===" % len(y_test))
log(results.round(3).to_string(index=False))
log(f"\nv1 training accuracy: {train_acc:.3f}   v1 test accuracy: {test_acc:.3f}   gap: {train_acc - test_acc:+.3f}")

# ---------------------------------------------------------------- 4. decision rule
def decide(appt: dict) -> str:
    """Overbook if v1 predicts a no-show, otherwise hold."""
    row = pd.DataFrame([appt])[FEATURES]
    return "overbook" if v1.predict(row)[0] == 1 else "hold"

def leaf_path(appt: dict) -> str:
    row = pd.DataFrame([appt])[FEATURES]
    node_ids = v1.decision_path(row).indices
    t = v1.tree_
    steps = []
    for n in node_ids:
        if t.children_left[n] == -1:  # leaf
            show, ns = t.value[n][0] * t.n_node_samples[n]
            steps.append(f"LEAF #{n}: {int(round(ns))} no-show / {int(round(show))} show "
                         f"in training ({ns / (ns + show):.0%} no-show)")
        else:
            f, thr = FEATURES[t.feature[n]], t.threshold[n]
            side = "<=" if row.iloc[0][f] <= thr else ">"
            steps.append(f"{f} {side} {thr:.2f}")
    return "  ->  ".join(steps)

D = dict(lead_time_days=21, prior_no_shows=1, age=27, distance_miles=9.5,
         sms_reminder=0, new_patient=1, appointment_hour=15)
E = dict(lead_time_days=3, prior_no_shows=0, age=61, distance_miles=2.0,
         sms_reminder=1, new_patient=0, appointment_hour=9)

log("\n=== 4. DECISIONS ===")
for name, appt in [("D", D), ("E", E)]:
    log(f"{name}: {leaf_path(appt)}")
    log(f"   decision: {decide(appt).upper()}")

# ---------------------------------------------------------------- 5. value
def value(r):
    return r.TN * VALUE_TN + r.FP * VALUE_FP + r.FN * VALUE_FN + r.TP * VALUE_TP

results["total_value"] = results.apply(value, axis=1)
results["per_appt"] = results["total_value"] / len(y_test)
results["per_year"] = results["per_appt"] * ANNUAL_VOLUME

b, v = results.iloc[0], results.iloc[1]
log("\n=== 5. VALUE ON TEST APPOINTMENTS ===")
log(results[["model", "TN", "FP", "FN", "TP", "total_value", "per_appt", "per_year"]]
    .round(2).to_string(index=False))
log(f"\nv1 saves ${v.per_appt - b.per_appt:.2f} per appointment vs baseline "
    f"({(v.total_value - b.total_value):,.0f} on {len(y_test)} test appts)")
log(f"At {ANNUAL_VOLUME:,} appointments/year: baseline ${b.per_year:,.0f}, v1 ${v.per_year:,.0f}, "
    f"saving ${v.per_year - b.per_year:,.0f}/yr")
log(f"Loss reduced by {1 - v.total_value / b.total_value:.0%}")

results.round(4).to_csv(OUT / "results.csv", index=False)

# ---------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (_, r) in zip(axes, results.iterrows()):
    m = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(m, cmap="Greys", vmin=0, vmax=m.max() * 1.6)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{m[i, j]}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["predict show\n(hold)", "predict no-show\n(overbook)"])
    ax.set_yticks([0, 1], ["came", "no-show"])
    ax.set_title(f"{r.model}: accuracy {r.accuracy:.0%}, recall {r.recall:.0%}")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bars = ax.bar(["Hold every slot\n(baseline)", "Overbook when\ntree says no-show (v1)"],
              results["per_appt"], color=["#999999", "#222222"])
for bar, val in zip(bars, results["per_appt"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f"${val:.2f}", ha="center", va="top",
            color="white", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Value per appointment ($)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "value_per_appt.png", dpi=200)
plt.close(fig)

(OUT / "results.txt").write_text("\n".join(lines) + "\n")
print(f"\nSaved: {OUT}/results.txt, results.csv, tree_v1.png, confusion_matrices.png, value_per_appt.png")
