# """
# 01_eda.py - Exploratory Data Analysis
# Project: Is AI Changing US Employment? (CS 7830, Kamala Joshi & Diwakar Dahal)
# Run: python src/analysis/eda.py   |   Figures: results/figures/01_*.png
# """

# # %% Setup
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())  # project folder
# FIG = ROOT / "results" / "figures"
# os.makedirs(FIG, exist_ok=True)

# pd.set_option("display.max_columns", 60)
# pd.set_option("display.width", 140)
# plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


# def save(name):
#     plt.savefig(FIG / f"01_{name}.png", dpi=200, bbox_inches="tight")


# def standardize(X):
#     X = np.asarray(X, dtype=float)
#     return (X - X.mean(axis=0)) / X.std(axis=0)


# def pca_numpy(Xs):
#     U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
#     ratio = S**2 / (S**2).sum()
#     return ratio, Vt, Xs @ Vt.T


# def heatmap(ax, corr, labels=None, annot=False):
#     labels = list(corr.columns) if labels is None else labels
#     im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
#     ax.set_xticks(range(len(labels)), labels, rotation=90)
#     ax.set_yticks(range(len(labels)), labels)
#     ax.grid(False)
#     if annot:
#         for i in range(corr.shape[0]):
#             for j in range(corr.shape[1]):
#                 ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
#     plt.colorbar(im, ax=ax, shrink=0.7)


# def weighted_group_table(data, col, wcol):
#     d = data.dropna(subset=[col, wcol])
#     return d.groupby("exposure_group", observed=True).apply(
#         lambda g: pd.Series({"n": len(g), "mean": g[col].mean(),
#                              "weighted_mean": np.average(g[col], weights=g[wcol])}),
#         include_groups=False).round(3)


# # %% Load and prepare
# df = pd.read_csv(ROOT / "data" / "processed" / "occupations.csv", dtype={"soc_code": str})
# skill_cols = [c for c in df.columns if c.startswith("skill_")]

# FAMILY = {"11": "Management", "13": "Business & financial", "15": "Computer & math",
#           "17": "Architecture & eng.", "19": "Science", "21": "Community & social", "23": "Legal",
#           "25": "Education", "27": "Arts & media", "29": "Healthcare practitioners",
#           "31": "Healthcare support", "33": "Protective service", "35": "Food prep & serving",
#           "37": "Building & grounds", "39": "Personal care", "41": "Sales", "43": "Office & admin",
#           "45": "Farming", "47": "Construction", "49": "Installation & repair", "51": "Production",
#           "53": "Transportation"}
# df["family_name"] = df["soc_code"].str[:2].map(FAMILY)

# # BLS top-codes high wages (2022 cap $239,200; 2016 cap $208,000)
# CAP_2022, CAP_2016 = 239_200, 208_000
# df["wage_2022_topcoded"] = df["wage_2022"].isna() & (df["wage_2025"] >= CAP_2022)
# df.loc[df["wage_2022_topcoded"], "wage_2022"] = CAP_2022
# df["wage_2016_topcoded"] = df["wage_2016"].isna() & df["emp_2016"].notna() & (df["wage_2025"] >= CAP_2022)
# df.loc[df["wage_2016_topcoded"], "wage_2016"] = CAP_2016

# df["log_wage_2022"] = np.log(df["wage_2022"])
# df["job_zone_round"] = df["job_zone"].round().astype(int)
# df["all_other"] = df["title"].str.contains("All Other")
# df["extreme_change"] = df["d_log_emp_2022_2025"].abs() > 0.7
# df["exposure_group"] = pd.cut(df["exposure"], bins=[-0.001, 0, 0.1, 0.3, 1],
#                               labels=["0", "0–0.1", "0.1–0.3", ">0.3"])

# print(df.shape)
# print("Top-coded wages filled  2022:", df["wage_2022_topcoded"].sum(), " 2016:", df["wage_2016_topcoded"].sum())
# print("'All Other' rows:", df["all_other"].sum(), " extreme swings:", df["extreme_change"].sum(),
#       " non-whole job zones:", (df["job_zone"] % 1 != 0).sum())

# key = ["exposure", "emp_2022", "emp_2025", "wage_2022", "wage_2025", "job_zone", "education_level",
#        "d_log_emp_2022_2025", "d_log_emp_2016_2019", "d_log_wage_2022_2025"]
# print(df[key].describe().T.round(3))

# # %% A. Exposure
# nz = df.loc[df["exposure"] > 0, "exposure"]
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].hist(df["exposure"], bins=40, color="#4C72B0")
# axes[0].set(title="Exposure, all occupations", xlabel="exposure", ylabel="occupations")
# axes[1].hist(nz, bins=40, color="#DD8452")
# axes[1].set(title="Exposure > 0 only", xlabel="exposure")
# plt.tight_layout(); save("exposure_hist"); plt.show()

# print(f"Share exactly 0: {(df['exposure'] == 0).mean():.3f} | non-zero: {len(nz)} | skew (non-zero): {nz.skew():.2f}")
# print(nz.describe().round(3))

# fam_exp = (df.groupby("family_name")["exposure"]
#            .agg(mean="mean", share_nonzero=lambda s: (s > 0).mean(), n="count").sort_values("mean"))
# fig, ax = plt.subplots(figsize=(8, 7))
# ax.barh(fam_exp.index, fam_exp["mean"], color="#4C72B0")
# ax.set(title="Average exposure by job family", xlabel="mean exposure")
# plt.tight_layout(); save("exposure_by_family"); plt.show()
# print(fam_exp.sort_values("mean", ascending=False).round(3))

# cols = ["soc_code", "title", "family_name", "exposure", "emp_2022"]
# print("Highest exposure:"); print(df.nlargest(15, "exposure")[cols])
# print("Lowest non-zero exposure:"); print(df[df["exposure"] > 0].nsmallest(15, "exposure")[cols])

# # %% B. Employment and wages
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes[0, 0].hist(df["emp_2022"].dropna(), bins=50, color="#4C72B0")
# axes[0, 0].set(title="Employment 2022", xlabel="workers")
# axes[0, 1].hist(np.log10(df["emp_2022"].dropna()), bins=50, color="#4C72B0")
# axes[0, 1].set(title="Employment 2022 (log10)", xlabel="log10(workers)")
# axes[1, 0].hist(df["wage_2022"].dropna(), bins=50, color="#55A868")
# axes[1, 0].set(title="Median wage 2022", xlabel="$ per year")
# axes[1, 1].hist(df["log_wage_2022"].dropna(), bins=50, color="#55A868")
# axes[1, 1].set(title="Median wage 2022 (log)", xlabel="log(wage)")
# plt.tight_layout(); save("emp_wage_hist"); plt.show()

# print("Skew emp_2022:", round(df["emp_2022"].skew(), 2), "| log:", round(np.log(df["emp_2022"]).skew(), 2))
# print("Skew wage_2022:", round(df["wage_2022"].skew(), 2), "| log:", round(df["log_wage_2022"].skew(), 2))
# print(f"Total workers 2022: {df['emp_2022'].sum():,.0f} | 2025: {df['emp_2025'].sum():,.0f}")

# outcomes = {"d_log_emp_2022_2025": "Employment change 2022→2025",
#             "d_log_emp_2016_2019": "Employment change 2016→2019 (placebo)",
#             "d_log_wage_2022_2025": "Wage change 2022→2025"}
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# for ax, (col, title) in zip(axes, outcomes.items()):
#     ax.hist(df[col].dropna(), bins=50, color="#8172B2")
#     ax.axvline(0, color="k", lw=1)
#     ax.set(title=title, xlabel="log change")
# plt.tight_layout(); save("outcome_hist"); plt.show()

# for col in outcomes:
#     s = df[col].dropna()
#     print(f"{col}: n={len(s)} mean={s.mean():.3f} median={s.median():.3f} sd={s.std():.3f} share>0={(s > 0).mean():.3f}")

# ext = ["title", "family_name", "emp_2022", "emp_2025", "exposure", "d_log_emp_2022_2025"]
# print("Biggest increases:"); print(df.nlargest(10, "d_log_emp_2022_2025")[ext])
# print("Biggest decreases:"); print(df.nsmallest(10, "d_log_emp_2022_2025")[ext])

# d = df.dropna(subset=["d_log_emp_2022_2025", "emp_2022"]).copy()
# d["size_group"] = pd.qcut(d["emp_2022"], 4, labels=["smallest 25%", "2nd", "3rd", "largest 25%"])
# print(d.groupby("size_group", observed=True)["d_log_emp_2022_2025"]
#       .agg(count="count", sd="std", mean_abs_change=lambda s: s.abs().mean()).round(3))

# # %% Exposure vs employment change
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# for ax, col, emp, title in zip(axes, ["d_log_emp_2022_2025", "d_log_emp_2016_2019"],
#                                ["emp_2022", "emp_2016"], ["2022→2025 (after ChatGPT)", "2016→2019 (placebo)"]):
#     d = df.dropna(subset=[col, emp])
#     ax.scatter(d["exposure"], d[col], s=d[emp] / d[emp].max() * 400 + 3, alpha=0.4, color="#4C72B0", edgecolor="none")
#     ax.axhline(0, color="k", lw=1)
#     ax.set(title=title, xlabel="exposure", ylabel="log change in employment")
# plt.tight_layout(); save("exposure_vs_change"); plt.show()

# # Compare both periods on the same occupations
# same = df.dropna(subset=["d_log_emp_2016_2019", "d_log_emp_2022_2025", "emp_2016", "emp_2022"])
# print(f"Same-sample occupations: {len(same)}")
# print("2022→2025 (same occupations):"); print(weighted_group_table(same, "d_log_emp_2022_2025", "emp_2022"))
# print("2016→2019 (same occupations):"); print(weighted_group_table(same, "d_log_emp_2016_2019", "emp_2016"))
# print("2022→2025 (all occupations):");  print(weighted_group_table(df, "d_log_emp_2022_2025", "emp_2022"))

# nopl = df[df["d_log_emp_2016_2019"].isna()].dropna(subset=["d_log_emp_2022_2025"])
# print(f"No placebo data: n={len(nopl)} mean change={nopl['d_log_emp_2022_2025'].mean():.3f} "
#       f"mean exposure={nopl['exposure'].mean():.3f} (all: {df['exposure'].mean():.3f})")

# print("Share of 2022 workers by exposure group:")
# print((df.groupby("exposure_group", observed=True)["emp_2022"].sum() / df["emp_2022"].sum()).round(3))
# print(df[df["exposure_group"] == ">0.3"].nlargest(5, "emp_2022")[["title", "exposure", "emp_2022"]])

# # %% Change by job family
# order = df.groupby("family_name")["d_log_emp_2022_2025"].median().sort_values().index
# groups = [df.loc[df["family_name"] == f, "d_log_emp_2022_2025"].dropna() for f in order]
# fig, ax = plt.subplots(figsize=(9, 8))
# ax.boxplot(groups, orientation="horizontal", tick_labels=list(order), patch_artist=True,
#            boxprops=dict(facecolor="#C4D7EA"), flierprops=dict(markersize=2))
# ax.axvline(0, color="k", lw=1)
# ax.set(title="Employment change 2022→2025 by job family", xlabel="log change")
# plt.tight_layout(); save("change_by_family"); plt.show()

# d = df.dropna(subset=["d_log_emp_2022_2025"])
# print(d.groupby("family_name").apply(
#     lambda g: pd.Series({"n": len(g), "mean_exposure": g["exposure"].mean(),
#                          "weighted_change": np.average(g["d_log_emp_2022_2025"], weights=g["emp_2022"])}),
#     include_groups=False).sort_values("weighted_change").round(3))

# d = df.dropna(subset=["d_log_wage_2022_2025"])
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.scatter(d["exposure"], d["d_log_wage_2022_2025"], s=8, alpha=0.5, color="#55A868")
# ax.set(title="Exposure vs wage change 2022→2025", xlabel="exposure", ylabel="log change in median wage")
# plt.tight_layout(); save("exposure_vs_wage_change"); plt.show()
# print("Corr exposure vs wage change:", round(d["exposure"].corr(d["d_log_wage_2022_2025"]), 3))

# # %% C. Education, job zone and correlations
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].hist(df["job_zone"], bins=np.arange(1.75, 5.5, 0.25), color="#C44E52")
# axes[0].set(title="Job zone", xlabel="job zone (2 = little prep, 5 = extensive)")
# axes[1].hist(df["education_level"].dropna(), bins=30, color="#C44E52")
# axes[1].set(title="Required education", xlabel="education level (2 = HS, 6 = bachelor's, 8 = master's, 11 = doctoral)")
# plt.tight_layout(); save("jobzone_education_hist"); plt.show()
# print(df["job_zone_round"].value_counts().sort_index())

# main_vars = ["exposure", "log_wage_2022", "job_zone", "education_level", "skill_programming", "skill_writing",
#              "skill_mathematics", "skill_repairing", "skill_social_perceptiveness", "d_log_emp_2022_2025"]
# corr_rows = df[main_vars].dropna()
# fig, ax = plt.subplots(figsize=(9, 7))
# heatmap(ax, corr_rows.corr(), annot=True)
# ax.set_title(f"Correlations: predictors and outcome (n = {len(corr_rows)})")
# plt.tight_layout(); save("corr_main"); plt.show()

# X = df[["exposure", "log_wage_2022", "education_level", "job_zone"]].dropna()
# vif = {}
# for col in X.columns:
#     y = X[col].values
#     others = np.column_stack([np.ones(len(X)), X.drop(columns=col).values])
#     coef, *_ = np.linalg.lstsq(others, y, rcond=None)
#     r2 = 1 - ((y - others @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
#     vif[col] = 1 / (1 - r2)
# print(pd.Series(vif, name="VIF").round(2))

# # %% Skills and PCA
# skills = df[skill_cols].dropna()
# short = [c.replace("skill_", "").replace("_", " ") for c in skill_cols]
# skill_corr = skills.corr()

# fig, ax = plt.subplots(figsize=(13, 11))
# heatmap(ax, skill_corr, labels=short)
# ax.set_title("Correlations among the 35 O*NET skills")
# plt.tight_layout(); save("corr_skills"); plt.show()

# upper = skill_corr.where(np.triu(np.ones(skill_corr.shape, dtype=bool), k=1)).stack()
# print(f"Mean |r| between skills: {upper.abs().mean():.2f} | pairs |r|>0.8: {(upper.abs() > 0.8).sum()} of {len(upper)}")
# print(upper.abs().sort_values(ascending=False).head(10).round(3))

# ratio, components, pc_scores = pca_numpy(standardize(skills))
# cum = ratio.cumsum()
# k = np.arange(1, 16)
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.bar(k, ratio[:15], color="#4C72B0", label="per component")
# ax.plot(k, cum[:15], "o-", color="#C44E52", label="cumulative")
# ax.axhline(0.8, ls="--", color="grey"); ax.axhline(0.9, ls="--", color="grey")
# ax.set(title="PCA on skills: explained variance", xlabel="component", ylabel="share of variance", xticks=k)
# ax.legend()
# plt.tight_layout(); save("pca_scree"); plt.show()
# print(pd.DataFrame({"component": k, "variance": ratio[:15], "cumulative": cum[:15]}).round(3).to_string(index=False))
# print("Components for 80%:", int(np.argmax(cum >= 0.8) + 1), "| for 90%:", int(np.argmax(cum >= 0.9) + 1))

# loadings = pd.DataFrame(components[:3].T, index=short, columns=["PC1", "PC2", "PC3"])
# for pc in loadings:
#     s = loadings[pc].sort_values()
#     print(f"{pc} negative:", ", ".join(f"{i} ({v:.2f})" for i, v in s.head(5).items()))
#     print(f"{pc} positive:", ", ".join(f"{i} ({v:.2f})" for i, v in s.tail(5)[::-1].items()))

# fig, ax = plt.subplots(figsize=(7, 10))
# loadings[["PC1", "PC2"]].sort_values("PC2").plot.barh(ax=ax)
# ax.set(title="Skill loadings on PC1 and PC2", xlabel="loading")
# plt.tight_layout(); save("pca_loadings"); plt.show()

# pcs = pd.DataFrame(pc_scores[:, :4], index=skills.index, columns=["PC1", "PC2", "PC3", "PC4"])
# plot_df = df.join(pcs, how="inner")
# fig, ax = plt.subplots(figsize=(9, 7))
# sc = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=plot_df["exposure"], cmap="viridis", s=14, alpha=0.8)
# plt.colorbar(sc, label="exposure")
# for _, r in plot_df.nlargest(6, "exposure").iterrows():
#     ax.annotate(r["title"][:28], (r["PC1"], r["PC2"]), fontsize=7)
# ax.set(title="Occupations on the first two skill components", xlabel="PC1", ylabel="PC2")
# plt.tight_layout(); save("pca_scatter_exposure"); plt.show()

# print("Skill components vs controls:")
# print(pd.DataFrame({c: pcs.corrwith(plot_df[c]) for c in
#                     ["job_zone", "education_level", "log_wage_2022", "exposure"]}).round(2))

# # %% D. Missing values
# miss = df[key + skill_cols + ["wage_2016", "wage_2019", "emp_2016", "emp_2019"]].isna().sum()
# miss = miss[miss > 0].sort_values(ascending=False)
# fig, ax = plt.subplots(figsize=(8, 9))
# ax.barh(miss.index, miss.values, color="#937860")
# ax.set(title="Missing values per column (out of 756)", xlabel="missing")
# plt.tight_layout(); save("missing_counts"); plt.show()
# print(miss.to_frame("missing"))

# df["edu_missing"] = df["education_level"].isna()
# print(df.groupby("family_name")["edu_missing"].agg(missing="sum", n="count", share="mean")
#       .sort_values("share", ascending=False).round(3))
# print(df.groupby("edu_missing")[["exposure", "wage_2022", "emp_2022", "job_zone", "d_log_emp_2022_2025"]].median().round(3))

# df["placebo_missing"] = df["d_log_emp_2016_2019"].isna()
# print(df.groupby("exposure_group", observed=True)["placebo_missing"].agg(n="count", share_missing="mean").round(3))
# print(df[df["placebo_missing"]].nlargest(12, "exposure")[["title", "exposure", "emp_2022"]])

# sets = {
#     "regression with education": ["d_log_emp_2022_2025", "exposure", "wage_2022", "education_level", "job_zone"],
#     "regression without education": ["d_log_emp_2022_2025", "exposure", "wage_2022", "job_zone"],
#     "clustering (+35 skills)": ["exposure", "wage_2022", "education_level", "job_zone"] + skill_cols,
#     "placebo with education": ["d_log_emp_2016_2019", "exposure", "wage_2016", "education_level", "job_zone"],
#     "placebo without education": ["d_log_emp_2016_2019", "exposure", "wage_2016", "job_zone"],
# }
# print(pd.Series({k: len(df.dropna(subset=v)) for k, v in sets.items()}, name="rows kept (of 756)"))

# # %% E. Scaling
# feat = ["exposure", "wage_2022", "education_level", "job_zone", "skill_programming", "skill_writing"]
# d = df[feat].dropna()
# print(d.agg(["min", "max", "std"]).T.round(2))

# rng = np.random.default_rng(42)
# i, j = rng.integers(0, len(d), 2000), rng.integers(0, len(d), 2000)


# def distance_share(X):
#     sq = (X.values[i] - X.values[j]) ** 2
#     return pd.Series(sq.sum(axis=0) / sq.sum(), index=X.columns)


# dist = pd.DataFrame({"unscaled": distance_share(d),
#                      "standardized": distance_share(pd.DataFrame(standardize(d), columns=feat))}).round(4)
# fig, ax = plt.subplots(figsize=(8, 4))
# dist.plot.bar(ax=ax, rot=20)
# ax.set(title="Share of pairwise distance from each feature", ylabel="share")
# plt.tight_layout(); save("scaling_distance_share"); plt.show()
# print(dist)

# w = df["wage_2022"].dropna()
# fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
# axes[0].hist(w, bins=40); axes[0].set_title("wage")
# axes[1].hist(standardize(w.to_frame()).ravel(), bins=40); axes[1].set_title("standardized wage")
# axes[2].hist(standardize(np.log(w).to_frame()).ravel(), bins=40); axes[2].set_title("standardized log(wage)")
# plt.tight_layout(); save("wage_transform"); plt.show()

"""
eda.py - Exploratory Data Analysis
Project: Is AI Changing US Employment? (CS 7830, Kamala Joshi & Diwakar Dahal)
Run: python src/analysis/eda.py
Output: results/figures/01_*.png  (each chart also opens in a window: use the toolbar to zoom and pan)
Requires: pandas, numpy, matplotlib
"""

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)
SHOW = True   # set to False to only save the PNGs without opening windows

# %% Style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BEFORE, AFTER = BLUE, ORANGE          # 2016-19 is always blue, 2022-25 always orange
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ee8a89", "#b42d2c"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100,
})


def finish(fig, name, title, note):
    """Add a headline (the finding) and a 'how to read it' note, then save and show."""
    w, h = fig.get_size_inches()
    t_lines = textwrap.wrap(title, int(w * 8.5))
    n_lines = textwrap.wrap(note, int(w * 12.5))
    t_h, n_h = 0.24 * len(t_lines), 0.18 * len(n_lines)
    fig.tight_layout(rect=[0, 0, 1, 1 - (0.3 + t_h + n_h) / h])
    fig.text(0.01, 1 - 0.12 / h, "\n".join(t_lines), ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1 - (0.2 + t_h) / h, "\n".join(n_lines), ha="left", va="top", fontsize=9.5, color=INK2)
    fig.savefig(FIG / f"01_{name}.png", dpi=200)
    plt.show() if SHOW else plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}", horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -11),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK2)


def heatmap(ax, corr, annot=True, small=False):
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=60 if small else 35, ha="right",
                  fontsize=7 if small else 9)
    ax.set_yticks(range(corr.shape[0]), corr.index, fontsize=7 if small else 9)
    ax.grid(False)
    if annot:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else INK)
    plt.colorbar(im, ax=ax, shrink=0.8, label="correlation (r)")


def standardize(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def wmean(g, col, w):
    return np.average(g[col], weights=g[w])


# %% Load and prepare
df = pd.read_csv(ROOT / "data" / "processed" / "occupations.csv", dtype={"soc_code": str})
skill_cols = [c for c in df.columns if c.startswith("skill_")]
short = {c: c.replace("skill_", "").replace("_", " ") for c in skill_cols}

FAMILY = {"11": "Management", "13": "Business & financial", "15": "Computer & math",
          "17": "Architecture & eng.", "19": "Science", "21": "Community & social", "23": "Legal",
          "25": "Education", "27": "Arts & media", "29": "Healthcare practitioners",
          "31": "Healthcare support", "33": "Protective service", "35": "Food prep & serving",
          "37": "Building & grounds", "39": "Personal care", "41": "Sales", "43": "Office & admin",
          "45": "Farming", "47": "Construction", "49": "Installation & repair", "51": "Production",
          "53": "Transportation"}
df["family"] = df["soc_code"].str[:2].map(FAMILY)

# BLS top-codes high wages (2022 cap $239,200; 2016 cap $208,000)
CAP_2022, CAP_2016 = 239_200, 208_000
df["wage_2022_topcoded"] = df["wage_2022"].isna() & (df["wage_2025"] >= CAP_2022)
df.loc[df["wage_2022_topcoded"], "wage_2022"] = CAP_2022
df["wage_2016_topcoded"] = df["wage_2016"].isna() & df["emp_2016"].notna() & (df["wage_2025"] >= CAP_2022)
df.loc[df["wage_2016_topcoded"], "wage_2016"] = CAP_2016

df["log_wage_2022"] = np.log(df["wage_2022"])
df["job_zone_round"] = df["job_zone"].round().astype(int)
df["all_other"] = df["title"].str.contains("All Other")
df["extreme_change"] = df["d_log_emp_2022_2025"].abs() > 0.7
GROUPS = ["0", "0–0.1", "0.1–0.3", ">0.3"]
df["exposure_group"] = pd.cut(df["exposure"], bins=[-0.001, 0, 0.1, 0.3, 1], labels=GROUPS)
df["chg_main"] = 100 * df["d_log_emp_2022_2025"]   # log points x100, roughly %
df["chg_pre"] = 100 * df["d_log_emp_2016_2019"]

n = len(df)
print(df.shape)
print("Top-coded wages filled  2022:", df["wage_2022_topcoded"].sum(), " 2016:", df["wage_2016_topcoded"].sum())
print("'All Other' rows:", df["all_other"].sum(), " extreme swings:", df["extreme_change"].sum())

# %% A. Exposure
nz = df[df["exposure"] > 0]
n_zero = n - len(nz)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2.3]})
bars = a1.bar(["Exactly 0", "Above 0"], [n_zero, len(nz)], color=[INK2, BLUE], width=0.6)
for b, v in zip(bars, [n_zero, len(nz)]):
    a1.annotate(f"{v} ({v / n:.0%})", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9)
a1.set(title="Exactly 0 vs above 0", ylabel="occupations")
a1.grid(axis="x", visible=False)
a2.hist(nz["exposure"], bins=40, color=BLUE, edgecolor=SURFACE)
a2.axvline(nz["exposure"].median(), color=INK, ls="--", lw=1)
a2.annotate(f"median {nz['exposure'].median():.2f}", (nz["exposure"].median(), a2.get_ylim()[1] * 0.9),
            xytext=(5, 0), textcoords="offset points", fontsize=9)
a2.set(title="Exposure among occupations above 0", xlabel="exposure", ylabel="occupations")
finish(fig, "exposure_hist", f"{n_zero / n:.0%} of occupations have zero AI exposure; the rest are skewed toward low values",
       "Left: occupations that never appear in AI use vs those that do. Right: the spread among those that do. "
       "The large block of zeros is why we consider a yes/no flag alongside the exposure number.")

fam = (df.groupby("family").agg(mean_exp=("exposure", "mean"), share_nz=("exposure", lambda s: (s > 0).mean()))
       .sort_values("mean_exp"))
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(fam.index, fam["mean_exp"], color=BLUE, height=0.7)
for b, s in zip(bars, fam["share_nz"]):
    ax.annotate(f"{b.get_width():.3f}  ({s:.0%} of jobs use AI)", (b.get_width(), b.get_y() + b.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.set(xlabel="mean exposure", xlim=(0, fam["mean_exp"].max() * 1.45))
ax.grid(axis="y", visible=False)
finish(fig, "exposure_by_family", "Computer & math jobs are the most AI-exposed; hands-on families are near zero",
       "Average exposure in each job family. The percentage shows how many of the family's occupations use AI at all.")

top = df.nlargest(15, "exposure").iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top["title"].str.slice(0, 48), top["exposure"], color=BLUE, height=0.7)
label_bars(ax, bars, "{:.2f}", horizontal=True)
ax.set(xlabel="exposure")
ax.grid(axis="y", visible=False)
finish(fig, "exposure_top", "The most exposed jobs are text-, code- and information-heavy",
       "The 15 occupations with the highest exposure score.")

# %% B. Employment, wages and outcomes
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
panels = [(df["emp_2022"], "Employment 2022", "workers", BLUE), (np.log10(df["emp_2022"]), "Employment 2022 (log10)", "log10(workers)", BLUE),
          (df["wage_2022"], "Median wage 2022", "$ per year", AQUA), (df["log_wage_2022"], "Median wage 2022 (log)", "log(wage)", AQUA)]
for ax, (x, t, xl, c) in zip(axes.ravel(), panels):
    x = x.dropna()
    ax.hist(x, bins=50, color=c, edgecolor=SURFACE)
    ax.set(title=f"{t}   skew = {x.skew():.2f}", xlabel=xl, ylabel="occupations")
axes[0, 0].xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.1f}M" if v else "0")
finish(fig, "emp_wage_hist", "Employment and wages are heavily skewed; logs make them close to symmetric",
       "Left: raw values, where a few huge jobs and high salaries stretch the scale. Right: the same data after a log. "
       "Models use the log versions so a few extreme jobs don't dominate.")

fig, ax = plt.subplots(figsize=(11, 4.5))
edges = np.arange(-80, 84, 4)
for col, name, c, off in [("chg_pre", "2016→2019 (before ChatGPT)", BEFORE, -0.9), ("chg_main", "2022→2025 (after ChatGPT)", AFTER, 0.9)]:
    s = df[col].dropna()
    cnt, _ = np.histogram(s.clip(-80, 80), bins=edges)
    ax.bar(edges[:-1] + 2 + off, cnt / len(s) * 100, width=1.8, color=c, label=f"{name}: {(s > 0).mean():.0%} grew")
ax.axvline(0, color=INK, lw=1)
ax.set(xlabel="employment change (log points ×100 ≈ %, clipped at ±80)", ylabel="% of occupations")
ax.legend(loc="upper left")
finish(fig, "outcome_hist", f"Most occupations grew in both periods ({(df['d_log_emp_2022_2025'] > 0).mean():.0%} after ChatGPT)",
       "Distribution of each occupation's employment change. The grew/shrank split is the class balance for classification.")

d = df.dropna(subset=["d_log_emp_2022_2025", "emp_2022"]).copy()
d["size"] = pd.qcut(d["emp_2022"], 4, labels=["Smallest 25%", "2nd", "3rd", "Largest 25%"])
sz = d.groupby("size", observed=True).agg(swing=("chg_main", lambda s: s.abs().mean()), lo=("emp_2022", "min"), hi=("emp_2022", "max"))
fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.bar([f"{i}\n({lo:,.0f}–{hi:,.0f} workers)" for i, lo, hi in zip(sz.index, sz["lo"], sz["hi"])],
              sz["swing"], color=BLUE, width=0.6)
label_bars(ax, bars, "{:.1f}%")
ax.set(ylabel="average absolute change, %")
ax.grid(axis="x", visible=False)
finish(fig, "swing_by_size", "Small occupations swing far more than large ones",
       "Occupations split into four equal groups by 2022 size. Small jobs have noisy counts, so results should be weighted by employment.")

ext = pd.concat([df.nsmallest(10, "d_log_emp_2022_2025"), df.nlargest(10, "d_log_emp_2022_2025")]).sort_values("chg_main")
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh([f"{t[:42]}  ({e:,.0f})" for t, e in zip(ext["title"], ext["emp_2022"])], ext["chg_main"],
               color=np.where(ext["chg_main"] > 0, BLUE, ORANGE), height=0.7)
ax.axvline(0, color=INK, lw=1)
ax.set(xlabel="employment change 2022→2025, log points ×100")
ax.grid(axis="y", visible=False)
finish(fig, "biggest_changes", "The biggest gainers and losers are mostly small jobs",
       "Ten largest increases and decreases; 2022 workers in brackets. Most are tiny occupations or job-code changes "
       "(e.g. taxi drivers) rather than AI effects.")

# %% C. Exposure vs employment change (research question)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (col, emp, c, t) in zip(axes, [("chg_pre", "emp_2016", BEFORE, "2016→2019 (before ChatGPT)"),
                                       ("chg_main", "emp_2022", AFTER, "2022→2025 (after ChatGPT)")]):
    d = df.dropna(subset=[col, emp])
    ax.scatter(d["exposure"], d[col].clip(-100, 100), s=d[emp] / 8000 + 4, color=c, alpha=0.35, linewidths=0)
    gm = d.groupby("exposure_group", observed=True).agg(x=("exposure", "mean"), y=(col, "mean"))
    ax.plot(gm["x"], gm["y"], "o-", color=INK, lw=2, ms=6, label="average per exposure group")
    ax.axhline(0, color=INK2, lw=1)
    ax.set(title=t, xlabel="exposure", ylim=(-60, 60))
axes[0].set_ylabel("employment change, log points ×100")
axes[0].legend(loc="lower left")
finish(fig, "exposure_vs_change", "Each dot is an occupation; the black line is the average for each exposure group",
       "Dot size = number of workers; the y-axis is cut at ±60 so the main cloud is readable. Compare the black line "
       "across the two periods: a line that slopes down only after ChatGPT would suggest a new AI effect.")

same = df.dropna(subset=["d_log_emp_2016_2019", "d_log_emp_2022_2025", "emp_2016", "emp_2022"])
tab = same.groupby("exposure_group", observed=True).apply(lambda g: pd.Series({
    "n": len(g), "pre": g["chg_pre"].mean(), "main": g["chg_main"].mean(),
    "pre_w": wmean(g, "chg_pre", "emp_2016"), "main_w": wmean(g, "chg_main", "emp_2022")}), include_groups=False)
print(f"Same-sample comparison ({len(same)} occupations):"); print(tab.round(2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(len(tab))
for ax, (a, b), t in zip(axes, [("pre", "main"), ("pre_w", "main_w")],
                         ["Every occupation counts equally", "Weighted by number of workers"]):
    for col, name, c, off in [(a, "2016→2019 (before)", BEFORE, -0.2), (b, "2022→2025 (after)", AFTER, 0.2)]:
        label_bars(ax, ax.bar(x + off, tab[col], 0.38, color=c, label=name))
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, [f"{g}\n(n={int(k)})" for g, k in zip(tab.index, tab["n"])])
    ax.set(title=t, xlabel="exposure group")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
axes[0].set_ylabel("average change, log points ×100")
axes[0].legend(loc="lower left")
finish(fig, "before_after_same_sample",
       f"On the same {len(same)} occupations, the most exposed group went from {tab.loc['>0.3', 'pre']:+.1f} "
       f"before ChatGPT to {tab.loc['>0.3', 'main']:+.1f} after",
       "The key EDA comparison. Both periods use the same occupations, so the difference isn't caused by jobs that only "
       "exist in one period. Left: the typical occupation. Right: the typical worker.")

share = df.groupby("exposure_group", observed=True)["emp_2022"].sum() / df["emp_2022"].sum() * 100
ticks = [f"exposure {g}\n\n" + "\n".join(df[df["exposure_group"] == g].nlargest(3, "emp_2022")["title"].str.slice(0, 30))
         for g in share.index]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(ticks, share, color=BLUE, width=0.6)
label_bars(ax, bars, "{:.0f}%")
ax.tick_params(axis="x", labelsize=8.5)
ax.set(ylabel="% of all 2022 workers")
ax.grid(axis="x", visible=False)
finish(fig, "workers_by_group",
       f"The >0.3 group has only {(df['exposure_group'] == '>0.3').sum()} occupations but {share['>0.3']:.0f}% of workers",
       "Share of all workers in each exposure group, with the three largest occupations of each group listed below its bar. "
       "A few very large occupations drive the worker-weighted results.")

# %% D. Change by job family
order = df.groupby("family")["chg_main"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(11, 8))
data = [df.loc[df["family"] == f, "chg_main"].dropna() for f in order]
ax.boxplot(data, orientation="horizontal", tick_labels=list(order), patch_artist=True, widths=0.6, showfliers=False,
           boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color=INK, lw=1.5),
           whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
rng = np.random.default_rng(0)
for i, s in enumerate(data, 1):
    ax.scatter(s.clip(-80, 80), i + rng.uniform(-0.2, 0.2, len(s)), s=6, color=BLUE, alpha=0.4, linewidths=0)
ax.axvline(0, color=INK, lw=1)
ax.set(xlabel="employment change 2022→2025, log points ×100", xlim=(-80, 80))
ax.grid(axis="y", visible=False)
finish(fig, "change_by_family", "Employment change varies a lot within every family",
       "Each dot is an occupation. Boxes show the middle 50% and the median line; families are sorted by median change.")

fw = df.dropna(subset=["d_log_emp_2022_2025"]).groupby("family").apply(lambda g: pd.Series({
    "exposure": g["exposure"].mean(), "change": wmean(g, "chg_main", "emp_2022"), "workers": g["emp_2022"].sum()}),
    include_groups=False)
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(fw["exposure"], fw["change"], s=fw["workers"] / 40000 + 20, color=BLUE, alpha=0.65, edgecolors=SURFACE, linewidths=1.5)
LABEL = {"Office & admin", "Computer & math", "Sales", "Farming", "Production", "Management", "Personal care",
         "Arts & media", "Legal", "Business & financial", "Education", "Science", "Transportation", "Construction"}
for f, r in fw[fw.index.isin(LABEL)].iterrows():
    ax.annotate(f, (r["exposure"], r["change"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
ax.axhline(0, color=INK, lw=1)
ax.set(xlabel="mean exposure of the family", ylabel="worker-weighted change 2022→2025, log points ×100")
finish(fig, "family_exposure_vs_change",
       "Office & admin is highly exposed and shrinking, but low-exposure Farming and Production also shrank",
       "Each bubble is a job family (size = workers; small families near the top left are unlabelled). If AI were the only force, bubbles would fall along a downward "
       "line; the scatter shows other forces too (post-COVID recovery, tech layoffs).")

d = df.dropna(subset=["d_log_wage_2022_2025"])
slope, icpt = np.polyfit(d["exposure"], 100 * d["d_log_wage_2022_2025"], 1)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.scatter(d["exposure"], 100 * d["d_log_wage_2022_2025"], s=12, color=AQUA, alpha=0.5, linewidths=0, label="occupation")
xs = np.array([0, d["exposure"].max()])
ax.plot(xs, icpt + slope * xs, color=INK, lw=2, label=f"linear trend ({slope / 10:+.1f} per 0.1 exposure)")
ax.set(xlabel="exposure", ylabel="median wage change 2022→2025, log points ×100", ylim=(-20, 45))
ax.legend(loc="upper right")
finish(fig, "exposure_vs_wage_change",
       f"More exposed jobs saw slightly smaller raises (r = {d['exposure'].corr(d['d_log_wage_2022_2025']):.2f})",
       "Wage growth for each occupation against exposure. The line is a simple linear trend.")

# %% E. Education, job zone and correlations
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.5]})
jz = df["job_zone_round"].value_counts().sort_index()
label_bars(a1, a1.bar(jz.index.astype(str), jz.values, color=BLUE, width=0.6), "{:.0f}")
a1.set(title="Occupations per job zone", xlabel="job zone (2 = little prep, 5 = extensive)", ylabel="occupations")
a1.grid(axis="x", visible=False)
d = df.dropna(subset=["education_level"])
a2.scatter(d["job_zone"] + rng.uniform(-0.15, 0.15, len(d)), d["education_level"], s=10, color=BLUE, alpha=0.4, linewidths=0)
a2.set(title="Education vs job zone (each dot = an occupation)", xlabel="job zone",
       ylabel="education level")
finish(fig, "jobzone_education",
       f"Education and job zone measure almost the same thing (r = {d['education_level'].corr(d['job_zone']):.2f})",
       f"Education scale: 2 = high school, 6 = bachelor's, 8 = master's, 11 = doctoral. Education is missing for "
       f"{df['education_level'].isna().sum()} occupations and job zone for none, so job zone is the better control.")

main_vars = {"exposure": "exposure", "log_wage_2022": "log wage", "job_zone": "job zone",
             "education_level": "education", "skill_programming": "programming", "skill_writing": "writing",
             "skill_mathematics": "mathematics", "skill_repairing": "repairing",
             "skill_social_perceptiveness": "social perceptiveness", "d_log_emp_2022_2025": "emp. change 22→25"}
rows = df[list(main_vars)].dropna()
fig, ax = plt.subplots(figsize=(10, 7.5))
heatmap(ax, rows.corr().rename(index=main_vars, columns=main_vars))
finish(fig, "corr_main",
       "Wage, job zone, education and cognitive skills move together; exposure is linked to them only moderately",
       f"Correlations on the {len(rows)} occupations with all values. Red = move together, blue = move in opposite "
       "directions. Strongly linked controls confuse a regression (multicollinearity).")

X = df[["exposure", "log_wage_2022", "education_level", "job_zone"]].dropna()
vif = {}
for col in X.columns:
    y = X[col].values
    others = np.column_stack([np.ones(len(X)), X.drop(columns=col).values])
    coef, *_ = np.linalg.lstsq(others, y, rcond=None)
    vif[col] = 1 / (((y - others @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum())
vif = pd.Series(vif).sort_values()
print("VIF:"); print(vif.round(2))
fig, ax = plt.subplots(figsize=(9, 3.6))
bars = ax.barh(vif.index, vif.values, color=[ORANGE if v > 5 else BLUE for v in vif.values], height=0.6)
label_bars(ax, bars, "{:.1f}", horizontal=True)
ax.axvline(5, color=INK2, ls="--", lw=1)
ax.set(xlabel="variance inflation factor (VIF); dashed line = warning level 5")
ax.grid(axis="y", visible=False)
finish(fig, "vif", "Education and job zone overlap too much to use together; exposure does not overlap with the controls",
       "VIF shows how well each variable is predicted by the others. Above 5 (orange) is a warning sign.")

# %% F. Skills and PCA
skills = df[skill_cols].dropna()
skill_corr = skills.corr().rename(index=short, columns=short)
upper = skill_corr.where(np.triu(np.ones(skill_corr.shape, dtype=bool), k=1)).stack()
fig, ax = plt.subplots(figsize=(12, 10))
heatmap(ax, skill_corr, annot=False, small=True)
finish(fig, "corr_skills", f"The 35 skills overlap heavily: {(upper.abs() > 0.8).sum()} pairs correlate above 0.8",
       f"Correlations among all skill pairs (average |r| = {upper.abs().mean():.2f}). Large red blocks are skills that "
       "measure nearly the same thing, which is why we compress them with PCA.")

Xs = standardize(skills)
U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
ratio = S**2 / (S**2).sum()
cum = ratio.cumsum()
k80, k90 = int(np.argmax(cum >= 0.8) + 1), int(np.argmax(cum >= 0.9) + 1)
pcs = pd.DataFrame(Xs @ Vt[:4].T, index=skills.index, columns=["PC1", "PC2", "PC3", "PC4"])
loadings = pd.DataFrame(Vt[:4].T, index=[short[c] for c in skill_cols], columns=pcs.columns)

k = np.arange(1, 16)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(k, ratio[:15] * 100, color=BLUE, width=0.7, label="each component")
ax.plot(k, cum[:15] * 100, "o-", color=ORANGE, lw=2, ms=5, label="cumulative")
for lvl in (80, 90):
    ax.axhline(lvl, color=INK2, ls=":", lw=1)
    ax.text(15.6, lvl, f"{lvl}%", va="center", fontsize=8.5, color=INK2)
ax.annotate(f"{k80} components = {cum[k80 - 1]:.1%}", (k80, cum[k80 - 1] * 100), xytext=(15, -30),
            textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))
ax.set(xticks=k, xlabel="component", ylabel="% of skill variance explained")
ax.legend(loc="center right")
ax.grid(axis="x", visible=False)
finish(fig, "pca_scree", f"{k80} components keep {cum[k80 - 1]:.0%} of the information in 35 skills ({k90} for 90%)",
       f"Bars: how much of the skill variation each component captures. Line: running total. PC1 alone captures {ratio[0]:.0%}.")

lo = loadings.sort_values("PC1")
fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
for ax, pc, t in zip(axes, ["PC1", "PC2"], ["PC1: overall skill demand", "PC2: people work vs machine work"]):
    ax.barh(lo.index, lo[pc], color=np.where(lo[pc] > 0, ORANGE, BLUE), height=0.7)
    ax.axvline(0, color=INK, lw=1)
    ax.set(title=t, xlabel="weight (loading)")
    ax.grid(axis="y", visible=False)
axes[0].tick_params(axis="y", labelsize=8.5)
finish(fig, "pca_loadings", "PC1 measures how much thinking and communication a job needs",
       "Each bar is a skill's weight in the component. In PC1 almost every cognitive skill pulls the same way, so it works "
       "as a 'skill level' score. PC2 separates service and social skills from equipment and quality-control skills. "
       "The sign of a component is arbitrary; only the pattern matters.")

pdf = df.join(pcs, how="inner")
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(pdf["PC1"], pdf["PC2"], c=pdf["exposure"], cmap=SEQUENTIAL, s=18, linewidths=0)
plt.colorbar(sc, ax=ax, label="exposure (darker = more AI)")
picks = pd.concat([pdf.nlargest(3, "exposure"), pdf.nsmallest(1, "PC1"), pdf.nlargest(1, "PC1"), pdf.nsmallest(1, "PC2")])
for k, (_, r) in enumerate(picks.drop_duplicates("title").iterrows()):
    right = r["PC1"] > 2
    ax.annotate(r["title"][:30], (r["PC1"], r["PC2"]), xytext=(-12 if right else 12, 14 if k % 2 else -14),
                textcoords="offset points", fontsize=8, color=INK, ha="right" if right else "left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.set(xlabel="PC1  (← more thinking and communication | less →)", ylabel="PC2  (← machine work | people work →)")
finish(fig, "pca_scatter_exposure", "AI-exposed jobs sit in the high-skill, people-oriented corner",
       "Each dot is an occupation placed by its skill profile and coloured by exposure. Labelled: the most exposed jobs "
       "and the extremes of each component.")

pc_ctrl = pd.DataFrame({c: pcs.corrwith(pdf[c]) for c in ["job_zone", "education_level", "log_wage_2022", "exposure"]})
pc_ctrl.columns = ["job zone", "education", "log wage", "exposure"]
fig, ax = plt.subplots(figsize=(8, 3.8))
heatmap(ax, pc_ctrl)
finish(fig, "pca_vs_controls",
       f"PC1 overlaps job zone (r = {pc_ctrl.loc['PC1', 'job zone']:.2f}), so don't use both in one regression",
       "Correlation of each skill component with the controls. PC2–PC4 add information the controls don't have.")

# %% G. Missing values
miss_cols = ["education_level", "wage_2016", "wage_2019", "emp_2016", "emp_2019", "d_log_emp_2016_2019",
             "d_log_wage_2022_2025", "wage_2022", "wage_2025", "d_log_emp_2022_2025"]
miss = df[miss_cols].isna().sum()
miss["any skill"] = df[skill_cols].isna().any(axis=1).sum()
miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
label_bars(ax, ax.barh(miss.index, miss.values, color=BLUE, height=0.7), "{:.0f}", horizontal=True)
ax.set(xlabel=f"missing values (out of {n})")
ax.grid(axis="y", visible=False)
finish(fig, "missing_counts", "Education and the 2016–2019 data have the most gaps",
       "2016/2019 gaps come from the 2018 job-code change: some occupations didn't exist before. "
       "Top-coded 2022 wages have already been filled.")

nopl = df[df["d_log_emp_2016_2019"].isna()]
pm = df.assign(m=df["d_log_emp_2016_2019"].isna()).groupby("exposure_group", observed=True)["m"].agg(["mean", "sum", "size"])
fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.bar(pm.index.astype(str), pm["mean"] * 100, color=BLUE, width=0.6)
for b, (_, r) in zip(bars, pm.iterrows()):
    ax.annotate(f"{r['mean']:.1%}\n({int(r['sum'])} of {int(r['size'])})", (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set(xlabel="exposure group", ylabel="% missing 2016→2019 change")
ax.margins(y=0.25)
ax.grid(axis="x", visible=False)
finish(fig, "placebo_missing",
       f"Missing placebo data rises with exposure; those jobs grew {nopl['chg_main'].mean():+.1f} on average after 2022",
       "Mostly new tech jobs (software developers, data scientists). This is why the main comparison uses only "
       "occupations that have both periods.")

sets = {
    "Regression, with education": ["d_log_emp_2022_2025", "exposure", "wage_2022", "education_level", "job_zone"],
    "Regression, without education": ["d_log_emp_2022_2025", "exposure", "wage_2022", "job_zone"],
    "Placebo, with education": ["d_log_emp_2016_2019", "exposure", "wage_2016", "education_level", "job_zone"],
    "Placebo, without education": ["d_log_emp_2016_2019", "exposure", "wage_2016", "job_zone"],
    "Clustering (+35 skills)": ["exposure", "wage_2022", "education_level", "job_zone"] + skill_cols,
}
kept = pd.Series({k: len(df.dropna(subset=v)) for k, v in sets.items()}).sort_values()
fig, ax = plt.subplots(figsize=(9, 3.8))
label_bars(ax, ax.barh(kept.index, kept.values, color=BLUE, height=0.6), "{:.0f}", horizontal=True)
ax.set(xlabel="rows kept if missing rows are dropped", xlim=(0, n * 1.1))
ax.grid(axis="y", visible=False)
finish(fig, "rows_kept", "Dropping education keeps over 100 more occupations",
       "How many occupations each planned feature set keeps if rows with blanks are dropped.")

# %% H. Scaling
feat = ["exposure", "wage_2022", "education_level", "job_zone", "skill_programming", "skill_writing"]
d = df[feat].dropna()
i, j = rng.integers(0, len(d), 2000), rng.integers(0, len(d), 2000)


def distance_share(X):
    sq = (np.asarray(X)[i] - np.asarray(X)[j]) ** 2
    return sq.sum(axis=0) / sq.sum() * 100


x = np.arange(len(feat))
fig, ax = plt.subplots(figsize=(10, 4.5))
label_bars(ax, ax.bar(x - 0.2, distance_share(d), 0.38, color=ORANGE, label="unscaled"), "{:.0f}%")
label_bars(ax, ax.bar(x + 0.2, distance_share(standardize(d)), 0.38, color=BLUE, label="standardized"), "{:.0f}%")
ax.set_xticks(x, feat)
ax.set(ylabel="% of distance between occupations")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
finish(fig, "scaling_distance_share", "Without scaling, wage alone decides which jobs look similar",
       "Share of the distance between random pairs of occupations that comes from each feature. Clustering uses these "
       "distances, so unscaled clusters would just be pay groups.")

w = df["wage_2022"].dropna()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, xv, t in zip(axes, [w, standardize(w), standardize(np.log(w))], ["wage", "standardized wage", "standardized log(wage)"]):
    xv = pd.Series(np.ravel(xv))
    ax.hist(xv, bins=40, color=BLUE, edgecolor=SURFACE)
    ax.set(title=f"{t}   skew = {xv.skew():.2f}")
axes[0].ticklabel_format(axis="x", style="plain")
finish(fig, "wage_transform", "Log first, then standardize: scaling alone keeps the skew",
       "Standardizing changes the units but not the shape (middle). Taking the log first gives a more symmetric feature (right).")

print("Saved figures to", FIG)