import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())   
sys.path.append(str(ROOT / "src"))                                                     
from features import build_features, REG_BASE, REG_FULL, PLACEBO_BASE

TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 170)
SHOW = True   

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False})

#  Data
df = build_features()
df["y_main"] = 100 * df["d_log_emp_2022_2025"]                       
df["y_pre"] = 100 * df["d_log_emp_2016_2019"]
df["y_wage"] = 100 * df["d_log_wage_2022_2025"]
df["y_wage_pre"] = 100 * (np.log(df["wage_2019"]) - np.log(df["wage_2016"]))
df["exp10"] = df["exposure"] * 10                                    
df["exp_group"] = pd.cut(df["exposure"], [-0.001, 0, 0.1, 0.3, 1], labels=["none", "low", "mid", "high"])


def fit(formula, data, weight=None, cluster=None):
    """OLS (or weighted OLS) with robust standard errors, or clustered by a column."""
    model = smf.wls(formula, data, weights=data[weight]) if weight else smf.ols(formula, data)
    if cluster is not None:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0]})
    return model.fit(cov_type="HC1")


def row(name, m, term="exp10", weight=None):
    lo, hi = m.conf_int().loc[term]
    return {"model": name, "weights": "workers" if weight else "jobs", "n": int(m.nobs),
            "effect": m.params[term], "ci_low": lo, "ci_high": hi, "p": m.pvalues[term], "r2": m.rsquared}


def rhs(features):
    return " + ".join("exp10" if f == "exposure" else f for f in features)


def show(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"02_{name}.png", dpi=200)
    plt.show() if SHOW else plt.close(fig)


def run_specs(specs):
    rows = []
    for name, y, feats, w in specs:
        data = df.dropna(subset=[y, w] + feats)
        for weight in [None, w]:
            rows.append(row(name, fit(f"{y} ~ {rhs(feats)}", data, weight), weight=weight))
    return pd.DataFrame(rows)


main = run_specs([("Employment 2022-25, base controls", "y_main", REG_BASE, "emp_2022"),
                  ("Employment 2022-25, full controls", "y_main", REG_FULL, "emp_2022"),
                  ("Placebo 2016-19, base controls", "y_pre", PLACEBO_BASE, "emp_2016")])
print("\n=== 1. Employment: exposure effect (per +0.1 exposure, log points x100 over 3 years) ===")
print(main.round(3).to_string(index=False))
main.to_csv(TAB / "regression_main.csv", index=False)

wage = run_specs([("Wages 2022-25, base controls", "y_wage", REG_BASE, "emp_2022"),
                  ("Wages 2022-25, full controls", "y_wage", REG_FULL, "emp_2022"),
                  ("Wage placebo 2016-19, base controls", "y_wage_pre", PLACEBO_BASE, "emp_2016")])
print("\n=== 2. Wages: exposure effect on median wage change (per +0.1 exposure, log points x100) ===")
print(wage.round(3).to_string(index=False))
wage.to_csv(TAB / "regression_wage.csv", index=False)

rows = []
for outcome, y in [("Employment", "y_main"), ("Wages", "y_wage")]:
    data = df.dropna(subset=[y, "emp_2022", "log_wage_2022", "job_zone"])
    for weight in [None, "emp_2022"]:
        m = fit(f"{y} ~ C(exp_group, Treatment('none')) + log_wage_2022 + job_zone", data, weight)
        for g in ["low", "mid", "high"]:
            term = f"C(exp_group, Treatment('none'))[T.{g}]"
            lo, hi = m.conf_int().loc[term]
            rows.append({"outcome": outcome, "weights": "workers" if weight else "jobs", "group": g,
                         "n_group": int((data["exp_group"] == g).sum()), "effect_vs_none": m.params[term],
                         "ci_low": lo, "ci_high": hi, "p": m.pvalues[term]})
groups = pd.DataFrame(rows)
print("\n=== 3. Exposure groups vs zero exposure (low 0-0.1, mid 0.1-0.3, high >0.3; log points x100) ===")
print(groups.round(3).to_string(index=False))
groups.to_csv(TAB / "regression_groups.csv", index=False)


both = df.dropna(subset=["y_pre", "y_main", "log_wage_2016", "log_wage_2022", "emp_2016", "emp_2022", "job_zone"])
stack = pd.concat([
    pd.DataFrame({"soc": both["soc_code"], "y": both["y_pre"], "lw": both["log_wage_2016"], "w": both["emp_2016"],
                  "exp10": both["exp10"], "job_zone": both["job_zone"], "family": both["family"], "after": 0}),
    pd.DataFrame({"soc": both["soc_code"], "y": both["y_main"], "lw": both["log_wage_2022"], "w": both["emp_2022"],
                  "exp10": both["exp10"], "job_zone": both["job_zone"], "family": both["family"], "after": 1}),
])
rows = []
for name, f in [("Change after ChatGPT", "y ~ exp10 * after + lw * after + job_zone * after"),
                ("Change after ChatGPT, + family", "y ~ exp10 * after + lw * after + job_zone * after + C(family) * after")]:
    for weight in [None, "w"]:
        m = fit(f, stack, weight, cluster="soc")
        r = row(name, m, "exp10:after", weight)
        r["n"] = len(both)                               
        r["effect_before"] = m.params["exp10"]
        rows.append(r)
change = pd.DataFrame(rows)
print(f"\n=== 4. Change in the exposure effect after ChatGPT ({len(both)} occupations, both periods) ===")
print("effect = (effect after) - (effect before); effect_before = slope in 2016-19")
print(change.round(3).to_string(index=False))
change.to_csv(TAB / "regression_change.csv", index=False)

base = df.dropna(subset=["y_main", "emp_2022"] + REG_BASE)
pre = base.dropna(subset=["y_pre"])                                  
checks = [
    ("Base model", base, f"y_main ~ {rhs(REG_BASE)}", "exp10", None),
    ("Without 'All Other' and extreme jobs", base[~base["all_other"] & ~base["extreme_change"]],
     f"y_main ~ {rhs(REG_BASE)}", "exp10", None),
    ("+ job family controls", base, f"y_main ~ {rhs(REG_BASE)} + C(family)", "exp10", None),
    ("Errors clustered by job family", base, f"y_main ~ {rhs(REG_BASE)}", "exp10", "family"),
    ("Same jobs as the trend check, no trend", pre, f"y_main ~ {rhs(REG_BASE)}", "exp10", None),
    ("+ 2016-19 trend as a control", pre, f"y_main ~ {rhs(REG_BASE)} + y_pre", "exp10", None),
    ("+ exposure x job zone (interaction)", base, f"y_main ~ {rhs(REG_BASE)} + exp10:job_zone", "exp10:job_zone", None),
]
rows = []
for name, data, f, term, cl in checks:
    for weight in [None, "emp_2022"]:
        rows.append(row(name, fit(f, data, weight, cluster=cl), term, weight))
robust = pd.DataFrame(rows)
print("\n=== 5. Robustness (employment 2022-25, base controls) ===")
print("for the interaction row, effect = how the exposure effect changes per +1 job zone")
print(robust.round(3).to_string(index=False))
robust.to_csv(TAB / "regression_robustness.csv", index=False)


m_diag = smf.ols(f"y_main ~ {rhs(REG_BASE)}", base).fit()
infl = m_diag.get_influence()
cooks = pd.Series(infl.cooks_distance[0], index=base.index)
cutoff = 4 / len(base)
bp_stat, bp_p, _, _ = het_breuschpagan(m_diag.resid, m_diag.model.exog)
print("\n=== 6. Diagnostics (employment 2022-25, base controls, each job) ===")
print(f"Breusch-Pagan test for uneven spread: p = {bp_p:.4f} "
      f"({'uneven spread present; robust standard errors are needed and used' if bp_p < 0.05 else 'no strong evidence of uneven spread'})")
print(f"Influential jobs (Cook's distance > 4/n = {cutoff:.4f}): {(cooks > cutoff).sum()}")
print(base.assign(cooks=cooks).nlargest(8, "cooks")[["title", "exposure", "y_main", "cooks"]].round(3).to_string())
no_infl = fit(f"y_main ~ {rhs(REG_BASE)}", base[cooks <= cutoff])
print(f"Exposure effect without influential jobs: {no_infl.params['exp10']:.3f} (p = {no_infl.pvalues['exp10']:.4f}, "
      f"n = {int(no_infl.nobs)}) vs {m_diag.params['exp10']:.3f} with them")

Xv = sm.add_constant(df[REG_FULL].dropna())
vif = pd.Series({c: 1 / (1 - sm.OLS(Xv[c], Xv.drop(columns=c)).fit().rsquared) for c in REG_FULL}, name="VIF")
print("VIF for the full feature set (above 5 = warning):")
print(vif.round(2).to_string())
pd.DataFrame({"breusch_pagan_p": [bp_p], "influential_jobs": [(cooks > cutoff).sum()],
              "effect_all": [m_diag.params["exp10"]], "effect_without_influential": [no_infl.params["exp10"]],
              "p_without_influential": [no_infl.pvalues["exp10"]]}).to_csv(TAB / "regression_diagnostics.csv", index=False)
vif.to_csv(TAB / "regression_vif.csv")

# %% Figures
def effect_chart(ax, table):
    t = table.iloc[::-1].reset_index(drop=True)
    colors = [BLUE if "lacebo" in m else ORANGE for m in t["model"]]
    ax.errorbar(t["effect"], t.index, xerr=[t["effect"] - t["ci_low"], t["ci_high"] - t["effect"]],
                fmt="none", ecolor=colors, elinewidth=2)
    ax.scatter(t["effect"], t.index, c=colors, s=55, zorder=3, edgecolors="white", linewidths=1.5)
    x_text = max(t["ci_high"].max(), 0) + 0.15
    for i, r in t.iterrows():
        p_text = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
        ax.text(x_text, i, f"{r['effect']:+.2f}  ({p_text})", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(t.index, t["model"].str.replace(", ", "\n") + " · each " + t["weights"].str.rstrip("s"), fontsize=8.5)
    ax.set_xlim(min(t["ci_low"].min(), 0) - 0.3, x_text + 1.0)
    ax.grid(axis="y", visible=False)


fig, ax = plt.subplots(figsize=(10, 4.8))
effect_chart(ax, main)
ax.set_xlabel("effect of +0.1 AI exposure on 3-year employment change (log points ×100), with 95% CI")
show(fig, "exposure_effect", "More AI exposure goes with slower employment growth after ChatGPT (orange) than before (blue)")

fig, ax = plt.subplots(figsize=(10, 4.8))
effect_chart(ax, wage)
ax.set_xlabel("effect of +0.1 AI exposure on 3-year median wage change (log points ×100), with 95% CI")
show(fig, "wage_effect", "Exposed jobs had slightly slower wage growth after ChatGPT (orange), but already did before (blue)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for ax, outcome in zip(axes, ["Employment", "Wages"]):
    g = groups[groups["outcome"] == outcome]
    x = np.arange(3)
    for k, (wt, c, off) in enumerate([("jobs", INK2, -0.15), ("workers", AQUA, 0.15)]):
        gg = g[g["weights"] == wt]
        ax.errorbar(x + off, gg["effect_vs_none"], yerr=[gg["effect_vs_none"] - gg["ci_low"], gg["ci_high"] - gg["effect_vs_none"]],
                    fmt="o", color=c, ms=7, elinewidth=2, mec="white", label=f"each {wt.rstrip('s')}")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, [f"{gr}\n(n={n})" for gr, n in zip(g["group"].unique(), g.drop_duplicates("group")["n_group"])])
    ax.set(title=f"{outcome} change vs zero-exposure jobs", xlabel="exposure group (low 0–0.1, mid 0.1–0.3, high >0.3)",
           ylabel="difference, log points ×100 (95% CI)")
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="lower left")
show(fig, "exposure_groups", "The employment gap is concentrated in the most exposed jobs")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fitted, resid = m_diag.fittedvalues, m_diag.resid
axes[0, 0].scatter(fitted, resid, s=8, color=BLUE, alpha=0.5)
axes[0, 0].axhline(0, color=INK, lw=1)
axes[0, 0].set(title="Residuals vs fitted (look for patterns)", xlabel="fitted value", ylabel="residual")
sm.qqplot(resid, line="45", fit=True, ax=axes[0, 1], markersize=3, color=BLUE)
axes[0, 1].get_lines()[1].set_color(ORANGE)
axes[0, 1].set_title("Q-Q plot (points on the line = normal residuals)")
axes[1, 0].stem(np.arange(len(cooks)), cooks.values, markerfmt=" ", basefmt=" ", linefmt=BLUE)
axes[1, 0].axhline(cutoff, color=ORANGE, ls="--", lw=1, label=f"4/n = {cutoff:.4f}")
for idx in cooks.nlargest(3).index:
    axes[1, 0].annotate(base.loc[idx, "title"][:28], (list(cooks.index).index(idx), cooks[idx]), xytext=(5, 0),
                        textcoords="offset points", fontsize=7.5)
axes[1, 0].set(title="Cook's distance (influential jobs above the line)", xlabel="occupation", ylabel="Cook's distance")
axes[1, 0].legend(loc="upper left")
axes[1, 1].hist(resid, bins=50, color=BLUE, edgecolor="white")
axes[1, 1].set(title="Residual distribution", xlabel="residual", ylabel="occupations")
show(fig, "diagnostics", "Regression diagnostics: uneven spread and heavy tails, so robust errors and checks are needed")

print("\nSaved tables to", TAB, "and figures to", FIG)