import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())  
sys.path.append(str(ROOT / "src"))                                                     
from features import build_features, REG_FULL

TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 170)
SHOW = True   

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False})


def show(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"03_{name}.png", dpi=200)
    plt.show() if SHOW else plt.close(fig)


# %% Data
df = build_features()
df["exp10"] = df["exposure"] * 10                                      
df["grew_pre"] = np.where(df["d_log_emp_2016_2019"].isna(), np.nan, (df["d_log_emp_2016_2019"] > 0).astype(float))
df.loc[df["d_log_emp_2016_2019"] == 0, "grew_pre"] = np.nan

FEATURES = REG_FULL                                                     
NO_EXPOSURE = [f for f in FEATURES if f not in ("exposure", "any_exposure")]
data = df.dropna(subset=["grew"] + FEATURES).reset_index(drop=True)
X, y = data[FEATURES], data["grew"].astype(int)
print(f"Occupations: {len(data)} | grew: {y.mean():.1%} | shrank: {1 - y.mean():.1%}")

def logit_row(name, formula, d):
    m = smf.logit(formula, d).fit(disp=0, cov_type="HC1")
    lo, hi = m.conf_int().loc["exp10"]
    return {"model": name, "n": int(m.nobs), "odds_ratio": np.exp(m.params["exp10"]), "ci_low": np.exp(lo),
            "ci_high": np.exp(hi), "p": m.pvalues["exp10"], "pseudo_r2": m.prsquared}


controls = "log_wage_2022 + job_zone"
pre = df.dropna(subset=["grew_pre", "log_wage_2016", "job_zone"])
rows = [
    logit_row("Grew 2022-25, base controls", f"grew ~ exp10 + {controls}", data),
    logit_row("Grew 2022-25, full controls", "grew ~ exp10 + any_exposure + log_wage_2022 + job_zone + pc2 + pc3 + pc4", data),
    logit_row("Placebo: grew 2016-19, base controls", "grew_pre ~ exp10 + log_wage_2016 + job_zone", pre),
]
near_zero = data[(data["d_log_emp_2022_2025"].abs() >= 0.02)]
rows.append(logit_row("Grew 2022-25, base, dropping |change| < 2%", f"grew ~ exp10 + {controls}", near_zero))
odds = pd.DataFrame(rows)
print("\n=== 1. Logistic regression: odds ratio of growing per +0.1 exposure (below 1 = less likely to grow) ===")
print(odds.round(3).to_string(index=False))
odds.to_csv(TAB / "classification_odds.csv", index=False)

both = df.dropna(subset=["grew", "grew_pre", "log_wage_2016", "log_wage_2022", "job_zone"])
stack = pd.concat([
    pd.DataFrame({"soc": both["soc_code"], "g": both["grew_pre"], "lw": both["log_wage_2016"],
                  "exp10": both["exp10"], "job_zone": both["job_zone"], "after": 0}),
    pd.DataFrame({"soc": both["soc_code"], "g": both["grew"], "lw": both["log_wage_2022"],
                  "exp10": both["exp10"], "job_zone": both["job_zone"], "after": 1})])
m = smf.logit("g ~ exp10 * after + lw * after + job_zone * after", stack).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(stack["soc"])[0]})
lo, hi = m.conf_int().loc["exp10:after"]
change = pd.DataFrame([{"occupations": len(both), "odds_ratio_before": np.exp(m.params["exp10"]),
                        "odds_ratio_after": np.exp(m.params["exp10"] + m.params["exp10:after"]),
                        "ratio_after_vs_before": np.exp(m.params["exp10:after"]), "ci_low": np.exp(lo),
                        "ci_high": np.exp(hi), "p": m.pvalues["exp10:after"]}])
print(f"\nDid the exposure effect change after ChatGPT? ({len(both)} occupations, both periods)")
print("ratio_after_vs_before below 1 = exposure became more negative for growth after 2022")
print(change.round(3).to_string(index=False))
change.to_csv(TAB / "classification_change.csv", index=False)

models = {
    "Baseline (always 'grew')": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, random_state=42),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
metrics = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1", "roc_auc": "roc_auc"}


def evaluate(feature_list):
    out = {}
    for name, model in models.items():
        s = cross_validate(model, data[feature_list], y, cv=cv, scoring=metrics)
        out[name] = {k: s[f"test_{k}"] for k in metrics}
    return out


with_exp = evaluate(FEATURES)
rows = [{"model": name, **{f"{k}": v.mean() for k, v in s.items()}, "auc_sd": s["roc_auc"].std()} for name, s in with_exp.items()]
scores = pd.DataFrame(rows)
print(f"\n=== 2. Cross-validated performance (5 folds x 3 repeats, {len(data)} occupations) ===")
print("balanced accuracy = average of the accuracy on grew and on shrank; ROC AUC 0.5 = guessing, 1 = perfect")
print(scores.round(3).to_string(index=False))
scores.to_csv(TAB / "classification_models.csv", index=False)

without_exp = evaluate(NO_EXPOSURE)
rows = []
for name in models:
    if name.startswith("Baseline"):
        continue
    diff = with_exp[name]["roc_auc"] - without_exp[name]["roc_auc"]         
    rows.append({"model": name, "auc_without_exposure": without_exp[name]["roc_auc"].mean(),
                 "auc_with_exposure": with_exp[name]["roc_auc"].mean(), "gain": diff.mean(),
                 "gain_sd": diff.std(), "folds_improved": f"{(diff > 0).sum()} of {len(diff)}"})
gain = pd.DataFrame(rows)
print("\n=== 3. ROC AUC with vs without the exposure features (same folds) ===")
print(gain.round(3).to_string(index=False))
gain.to_csv(TAB / "classification_exposure_gain.csv", index=False)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probs = {name: cross_val_predict(m, X, y, cv=folds, method="predict_proba")[:, 1] for name, m in models.items()}
best = max((n for n in models if not n.startswith("Baseline")), key=lambda n: scores.set_index("model").loc[n, "roc_auc"])
pred = (probs[best] >= 0.5).astype(int)
cm = confusion_matrix(y, pred)
print(f"\n=== 4. Best model by ROC AUC: {best} ===")
print(f"Out-of-fold accuracy {accuracy_score(y, pred):.3f}, F1 {f1_score(y, pred):.3f}, AUC {roc_auc_score(y, probs[best]):.3f}")
print(pd.DataFrame(cm, index=["actual shrank", "actual grew"], columns=["predicted shrank", "predicted grew"]))

data["exp_group"] = pd.cut(data["exposure"], [-0.001, 0, 0.1, 0.3, 1], labels=["none", "low", "mid", "high"])
by_group = data.assign(prob=probs[best]).groupby("exp_group", observed=True).agg(
    n=("grew", "size"), actual_share_grew=("grew", "mean"), predicted_chance=("prob", "mean"))
print("\nShare that grew vs model's average predicted chance, by exposure group:")
print(by_group.round(3).to_string())
by_group.to_csv(TAB / "classification_by_group.csv")

# Figures
fig, ax = plt.subplots(figsize=(10, 4))
t = odds.iloc[::-1].reset_index(drop=True)
colors = [BLUE if "Placebo" in m else ORANGE for m in t["model"]]
ax.errorbar(t["odds_ratio"], t.index, xerr=[t["odds_ratio"] - t["ci_low"], t["ci_high"] - t["odds_ratio"]],
            fmt="none", ecolor=colors, elinewidth=2)
ax.scatter(t["odds_ratio"], t.index, c=colors, s=55, zorder=3, edgecolors="white", linewidths=1.5)
x_text = t["ci_high"].max() + 0.02
for i, r in t.iterrows():
    p_text = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
    ax.text(x_text, i, f"{r['odds_ratio']:.2f}  ({p_text})", va="center", fontsize=8.5, color=INK2)
ax.axvline(1, color=INK, lw=1)
ax.set_yticks(t.index, t["model"].str.replace(", ", "\n", n=1), fontsize=8.5)
ax.set_xlim(t["ci_low"].min() - 0.03, x_text + 0.12)
ax.set_xlabel("odds ratio of growing per +0.1 AI exposure, with 95% CI (1 = no effect, below 1 = less likely to grow)")
ax.grid(axis="y", visible=False)
show(fig, "odds_ratios", "More exposed occupations were less likely to grow, after ChatGPT (orange) and already before (blue)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
s = scores.set_index("model")
ylab = s.index[::-1]
axes[0].barh(ylab, s.loc[ylab, "roc_auc"], xerr=s.loc[ylab, "auc_sd"], height=0.6,
             color=[INK2 if n.startswith("Baseline") else BLUE for n in ylab])
axes[0].axvline(0.5, color=INK, lw=1, ls="--")
axes[0].set(title="ROC AUC (0.5 = guessing)", xlim=(0.4, max(0.75, s["roc_auc"].max() + 0.05)), xlabel="ROC AUC (± sd across folds)")
for i, n in enumerate(ylab):
    axes[0].text(s.loc[n, "roc_auc"] + s.loc[n, "auc_sd"] + 0.005, i, f"{s.loc[n, 'roc_auc']:.3f}", va="center", fontsize=8.5, color=INK2)
w = 0.38
yy = np.arange(len(ylab))
axes[1].barh(yy + w / 2, s.loc[ylab, "accuracy"], w, color=BLUE, label="accuracy")
axes[1].barh(yy - w / 2, s.loc[ylab, "balanced_accuracy"], w, color=AQUA, label="balanced accuracy")
axes[1].set_yticks(yy, [])
axes[1].set(xlim=(0.4, 0.72), xlabel="share correct (baseline gets 60% by always saying 'grew')")
for i, n in enumerate(ylab):
    axes[1].text(s.loc[n, "accuracy"] + 0.004, i + w / 2, f"{s.loc[n, 'accuracy']:.3f}", va="center", fontsize=8, color=INK2)
    axes[1].text(s.loc[n, "balanced_accuracy"] + 0.004, i - w / 2, f"{s.loc[n, 'balanced_accuracy']:.3f}", va="center", fontsize=8, color=INK2)
axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1.06), ncol=2)
for ax in axes:
    ax.grid(axis="y", visible=False)
show(fig, "model_comparison", "Models beat guessing only modestly: growth vs decline is hard to predict")

fig, ax = plt.subplots(figsize=(8, 4))
yy = np.arange(len(gain))
ax.barh(yy + 0.2, gain["auc_without_exposure"], 0.38, color=INK2, label="without exposure")
ax.barh(yy - 0.2, gain["auc_with_exposure"], 0.38, color=ORANGE, label="with exposure")
for i, r in gain.iterrows():
    ax.text(r["auc_with_exposure"] + 0.003, i - 0.2, f"{r['gain']:+.3f} ({r['folds_improved']} folds better)",
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(yy, gain["model"])
ax.set(xlim=(0.5, 0.72), xlabel="ROC AUC (cross-validated)")
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
ax.grid(axis="y", visible=False)
show(fig, "exposure_gain", "Adding AI exposure barely changes how well the models predict growth vs decline")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 0.9, 1.1]})
for name, c in [("Logistic regression", BLUE), ("Random forest", AQUA), ("Gradient boosting", ORANGE)]:
    fpr, tpr, _ = roc_curve(y, probs[name])
    axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y, probs[name]):.2f})")
axes[0].plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, label="guessing (AUC 0.50)")
axes[0].set(title="ROC curves (out-of-fold)", xlabel="false positive rate", ylabel="true positive rate")
axes[0].legend(loc="lower right", fontsize=8.5)
axes[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%} of row)", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=10)
axes[1].set_xticks([0, 1], ["shrank", "grew"])
axes[1].set_yticks([0, 1], ["shrank", "grew"])
axes[1].set(title=f"Confusion matrix: {best}", xlabel="predicted", ylabel="actual")
axes[1].grid(False)
x = np.arange(len(by_group))
axes[2].bar(x - 0.2, by_group["actual_share_grew"] * 100, 0.38, color=INK2, label="actually grew")
axes[2].bar(x + 0.2, by_group["predicted_chance"] * 100, 0.38, color=ORANGE, label="model's predicted chance")
for i, r in enumerate(by_group.itertuples()):
    axes[2].text(i - 0.2, r.actual_share_grew * 100 + 1, f"{r.actual_share_grew:.0%}", ha="center", fontsize=8.5, color=INK2)
axes[2].set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(by_group.index, by_group["n"])])
axes[2].set(title="Chance of growing by exposure group", xlabel="exposure group (low 0–0.1, mid 0.1–0.3, high >0.3)",
            ylabel="%", ylim=(0, 85))
axes[2].legend(loc="upper right", fontsize=8.5)
axes[2].grid(axis="x", visible=False)
show(fig, "best_model", f"{best}: how well it separates growing from shrinking occupations")

print("\nSaved tables to", TAB, "and figures to", FIG)