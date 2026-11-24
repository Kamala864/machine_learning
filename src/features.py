from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          
DATA = ROOT / "data" / "processed" / "occupations.csv"

def load_data():
    df = pd.read_csv(DATA, dtype={"soc_code": str})
    df["family"] = df["soc_code"].str[:2]     
    return df


CAP_2022 = 239_200

def add_wage_features(df):
    hidden = df["wage_2022"].isna() & df["emp_2022"].notna() & (df["wage_2025"] >= CAP_2022)
    df["wage_2022_filled"] = hidden            # remember which rows we filled
    df.loc[hidden, "wage_2022"] = CAP_2022
    df["log_wage_2022"] = np.log(df["wage_2022"])   # natural log of wage_2022 (numpy)
    return df


def add_exposure_features(df):
    df["any_exposure"] = (df["exposure"] > 0).astype(int)    
    
    return df

def add_targets(df):
    df["grew"] = (df["d_log_emp_2022_2025"] > 0).astype(int)                 # classification target
    df.loc[df["d_log_emp_2022_2025"].isna(), "grew"] = np.nan                  # no change -> no label
    return df

def build_features():
    df = load_data()
    df = add_wage_features(df)
    df = add_exposure_features(df)
    df = add_skill_components(df)
    df = add_targets(df)
    return df

# Feature sets agreed from the EDA
# Regression: job zone as the skill/preparation control (not PC1, they overlap; not education, it overlaps
# with job zone and has 115 blanks). PC2-PC4 add skill information that job zone doesn't have.
REG_BASE = ["exposure", "log_wage_2022", "job_zone"]
REG_FULL = ["exposure", "any_exposure", "log_wage_2022", "job_zone", "skill_pc2", "skill_pc3", "skill_pc4"]


# Clustering: PC1 instead of job zone (same information, and using both would count it twice)
CLUSTER = ["exposure", "log_wage_2022", "skill_pc1", "skill_pc2", "skill_pc3", "skill_pc4"]

def model_table(df, target, features, weight="emp_2022"):
    """Rows that have the target, every feature and the weight. Returns X, y, w."""
    cols = list(dict.fromkeys([target] + features + [weight]))   # remove duplicates, keep order
    d = df.dropna(subset=cols)
    return d[features], d[target], d[weight]


def add_skill_components(df, n_components=4):
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    skills = df[skill_cols].dropna()                                   # PCA can't handle blanks

    z = (skills - skills.mean()) / skills.std(ddof=0)                  # standardize each skill
    U, S, Vt = np.linalg.svd(z.values, full_matrices=False)            # PCA via SVD
    explained = S**2 / (S**2).sum()                                    # share of variance per component
    scores = z.values @ Vt[:n_components].T    
    scores[:, 0] = -scores[:, 0]                        

    for i in range(n_components):
        df.loc[skills.index, f"skill_pc{i + 1}"] = scores[:, i]

    print(f"Rows used for PCA: {len(skills)} of {len(df)}")
    print(f"Variance kept by {n_components} components: {explained[:n_components].sum():.3f}")
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df["grew"].value_counts(dropna=False))

    sets = {
        "Regression, base": ("d_log_emp_2022_2025", REG_BASE),
        "Regression, full": ("d_log_emp_2022_2025", REG_FULL),
        "Placebo, base":    ("d_log_emp_2016_2019", REG_BASE),
        "Classification":   ("grew", REG_FULL),
    }
    for name, (target, feats) in sets.items():
        X, y, w = model_table(df, target, feats)
        print(f"{name:20s} rows: {len(X)}")

    X = df[CLUSTER].dropna()
    print(f"{'Clustering':20s} rows: {len(X)}")