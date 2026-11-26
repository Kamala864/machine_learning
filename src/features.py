from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "occupations.csv"


def load_data():
    df = pd.read_csv(DATA, dtype={"soc_code": str})
    df["family"] = df["soc_code"].str[:2]
    return df


CAP_2022, CAP_2016 = 239_200, 208_000


def add_wage_features(df):
    hidden = df["wage_2022"].isna() & df["emp_2022"].notna() & (df["wage_2025"] >= CAP_2022)
    df["wage_2022_filled"] = hidden            # remember which rows we filled
    df.loc[hidden, "wage_2022"] = CAP_2022
    df["log_wage_2022"] = np.log(df["wage_2022"])

    # placebo control: wage at the start of the placebo period
    hidden16 = df["wage_2016"].isna() & df["emp_2016"].notna() & (df["wage_2025"] >= CAP_2022)
    df["wage_2016_filled"] = hidden16
    df.loc[hidden16, "wage_2016"] = CAP_2016
    df["log_wage_2016"] = np.log(df["wage_2016"])
    return df


def add_exposure_features(df):
    df["any_exposure"] = (df["exposure"] > 0).astype(int)
    df["exposure_x_jobzone"] = df["exposure"] * df["job_zone"]   # optional: does AI hit low/high-skill jobs differently?
    return df


def add_skill_components(df, n_components=4):
    skill_cols = [c for c in df.columns if c.startswith("skill_")]   # PCs are named pc1.. so they never match this
    skills = df[skill_cols].dropna()                                   # PCA can't handle blanks

    z = (skills - skills.mean()) / skills.std(ddof=0)                  # standardize each skill
    U, S, Vt = np.linalg.svd(z.values, full_matrices=False)            # PCA via SVD
    explained = S**2 / (S**2).sum()                                    # share of variance per component
    scores = z.values @ Vt[:n_components].T

    # the sign of a component is arbitrary and can differ between computers, so fix it:
    # PC1 points toward more job preparation, the others toward more exposure
    jz = df.loc[skills.index, "job_zone"].values
    ex = df.loc[skills.index, "exposure"].values
    for i in range(n_components):
        ref = jz if i == 0 else ex
        if np.corrcoef(scores[:, i], ref)[0, 1] < 0:
            scores[:, i] = -scores[:, i]

    for i in range(n_components):
        df.loc[skills.index, f"pc{i + 1}"] = scores[:, i]

    print(f"Rows used for PCA: {len(skills)} of {len(df)}")
    print(f"Variance kept by {n_components} components: {explained[:n_components].sum():.3f}")
    return df


def add_flags(df):
    df["all_other"] = df["title"].str.contains("All Other")            # catch-all codes, for robustness checks
    df["extreme_change"] = df["d_log_emp_2022_2025"].abs() > 0.7       # roughly halved or doubled
    return df


def add_targets(df):
    df["grew"] = (df["d_log_emp_2022_2025"] > 0).astype(float)         # classification target
    df.loc[df["d_log_emp_2022_2025"].isna(), "grew"] = np.nan          # missing outcome -> no label
    df.loc[df["d_log_emp_2022_2025"] == 0, "grew"] = np.nan            # identical counts (2 jobs) -> no label
    return df


def build_features():
    df = load_data()
    df = add_wage_features(df)
    df = add_exposure_features(df)
    df = add_skill_components(df)
    df = add_flags(df)
    df = add_targets(df)
    return df


# Feature sets agreed from the EDA
# Regression: job zone as the skill/preparation control (not PC1, they overlap; not education, it overlaps
# with job zone and has 115 blanks). PC2-PC4 add skill information that job zone doesn't have.
REG_BASE = ["exposure", "log_wage_2022", "job_zone"]
REG_FULL = ["exposure", "any_exposure", "log_wage_2022", "job_zone", "pc2", "pc3", "pc4"]
PLACEBO_BASE = ["exposure", "log_wage_2016", "job_zone"]               # placebo uses the 2016 wage

# Clustering: PC1 instead of job zone (same information, and using both would count it twice).
# Standardize these before clustering: PC1 varies ~30x more than exposure.
CLUSTER = ["exposure", "log_wage_2022", "pc1", "pc2", "pc3", "pc4"]


def model_table(df, target, features, weight="emp_2022"):
    """Rows that have the target, every feature and the weight. Returns X, y, w."""
    cols = list(dict.fromkeys([target] + features + [weight]))   # remove duplicates, keep order
    d = df.dropna(subset=cols)
    return d[features], d[target], d[weight]


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df["grew"].value_counts(dropna=False))

    sets = {
        "Regression, base": ("d_log_emp_2022_2025", REG_BASE, "emp_2022"),
        "Regression, full": ("d_log_emp_2022_2025", REG_FULL, "emp_2022"),
        "Placebo, base":    ("d_log_emp_2016_2019", PLACEBO_BASE, "emp_2016"),
        "Classification":   ("grew", REG_FULL, "emp_2022"),
    }
    for name, (target, feats, weight) in sets.items():
        X, y, w = model_table(df, target, feats, weight)
        print(f"{name:20s} rows: {len(X)}")

    X = df[CLUSTER].dropna()
    print(f"{'Clustering':20s} rows: {len(X)}")