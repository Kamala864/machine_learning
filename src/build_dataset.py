"""

"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io
import re

RAW = Path("data/raw")         
OUT = Path("data/processed")   

OEWS_YEARS = [2016, 2019, 2022, 2025]           # BLS years we use
SOC_PATTERN = re.compile(r"^\d{2}-\d{4}$")      # valid code = 2 digits, dash, 4 digits

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(msg)


def load_exposure():
    df = pd.read_csv(RAW / "job_exposure.csv", dtype={"occ_code": str})
    df["soc_code"] = df["occ_code"].str.strip()


    # code that is NOT in the form XX-XXXX
    bad = ~df["soc_code"].str.match(SOC_PATTERN)
    if bad.any():
        log(f"[exposure] dropping {bad.sum()} rows with non-SOC codes: {df.loc[bad, 'soc_code'].tolist()[:10]}")


    # keeping those good rows & only the three columns we are using
    df = df.loc[~bad, ["soc_code", "title", "observed_exposure"]]
    # making it shorter
    df = df.rename(columns={"observed_exposure": "exposure"})

    duplicate_count = df["soc_code"].duplicated().sum()
    if duplicate_count:
        log(f"[exposure] {duplicate_count} duplicate codes found")
        df = df.drop_duplicates("soc_code")

    log(f"[exposure] occupations: {len(df)}")
    return df


def to_number(series):
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")   # "coerce" = turn non-numbers into NaN instead of crashing



def load_oews_year(year, cw):
    # Build the file name: str(2016)[2:] is "16" -> "oesm16nat.zip"
    zpath = RAW / f"oesm{str(year)[2:]}nat.zip"


    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith((".xlsx", ".xls")) and "national" in n.lower()
                 and "field" not in n.lower()]
        if not names:
            raise FileNotFoundError(f"No national_*.xlsx inside {zpath}: {z.namelist()}")
        with z.open(names[0]) as f:
            df = pd.read_excel(io.BytesIO(f.read()), dtype=str)

    df.columns = [c.strip().lower() for c in df.columns]

    group_col = "o_group" if "o_group" in df.columns else "occ_group"

    # Keep only detailed occupations
    df = df[df[group_col].str.strip().str.lower() == "detailed"].copy()

    emp = to_number(df["tot_emp"])      # tot_emp  = number of people employed
    wage = to_number(df["a_median"])    # a_median = median annual wage in dollars
    raw_wage = df["a_median"].astype(str).str.strip()
    log(f"[oews {year}] detailed occupations: {len(df)}; "
        f"employment missing/suppressed: {emp.isna().sum()}; "
        f"median wage missing/suppressed: {wage.isna().sum()} "
        f"(top-coded '#': {(raw_wage == '#').sum()})")

    out = pd.DataFrame({
        "soc_code": df["occ_code"].str.strip(),
        f"emp_{year}": emp.values,
        f"wage_{year}": wage.values,
    })

    # 2016 uses 2010 codes and 2019 a mix; translate them so they match the 2018 codes
    if year < 2021:
        out = convert_to_soc2018(out, year, cw)
    return out


def load_crosswalk():
    cw = pd.read_excel(RAW / "soc_2010_to_2018_crosswalk.xlsx", header=8, dtype=str)
    cw = cw.dropna(subset=["2010 SOC Code"])        # drop blank rows at the bottom
    cw = cw.rename(columns={"2010 SOC Code": "c10", "2018 SOC Code": "c18"})[["c10", "c18"]]
    return cw.apply(lambda s: s.str.strip())



def convert_to_soc2018(df, year, cw):
    split_10 = set(cw.groupby("c10")["c18"].nunique().loc[lambda s: s > 1].index)

    # Dictionaries for quick lookup: code -> value
    emp = dict(zip(df["soc_code"], df[f"emp_{year}"]))
    wage = dict(zip(df["soc_code"], df[f"wage_{year}"]))

    rows, direct, mapped = [], 0, 0


    for c18, group in cw.groupby("c18"):
        if year >= 2019 and c18 in emp:
            rows.append((c18, emp[c18], wage[c18]))
            direct += 1
            continue

        sources = list(group["c10"])

        if any(s in split_10 for s in sources):
            continue

        # Every source needs a value
        # then employment is the sum (one source = SAME, several = MERGED)
        values = [emp.get(s) for s in sources]
        if all(v is not None and not pd.isna(v) for v in values):
            w = wage.get(sources[0]) if len(sources) == 1 else np.nan
            rows.append((c18, sum(values), w))
            mapped += 1

    log(f"[oews {year}] converted to 2018 SOC: {direct} codes used directly, "
        f"{mapped} mapped through the crosswalk, {len(rows)} total")
    return pd.DataFrame(rows, columns=["soc_code", f"emp_{year}", f"wage_{year}"])



def load_oews(cw):
    # load all four BLS year and place them side by side
    merged = None

    for year in OEWS_YEARS:
        part = load_oews_year(year, cw)
        # keep an occupation even if it is missing in some years (those cells become NaN)
        merged = part if merged is None else merged.merge(part, on="soc_code", how="outer")
    log(f"[oews] occupations present in at least one year: {len(merged)}")
    all_years = merged[[f"emp_{y}" for y in OEWS_YEARS]].notna().all(axis=1).sum()
    log(f"[oews] occupations with employment in all four years: {all_years}")
    return merged


def read_onet(z, filename):
    match = [n for n in z.namelist() if n.endswith("/" + filename) or n == filename]
    if not match:
        raise FileNotFoundError(f"{filename} not found in O*NET zip")
    with z.open(match[0]) as f:
        return pd.read_excel(io.BytesIO(f.read()))


def onet_to_soc(code):
    return str(code)[:7]

def load_onet(skill_scale="LV"):
    with zipfile.ZipFile(RAW / "db_31_0_excel.zip") as z:
        skills = pd.concat([read_onet(z, "Essential Skills.xlsx"),
                            read_onet(z, "Transferable Skills.xlsx")], ignore_index=True)
        zones = read_onet(z, "Job Zones.xlsx")
        ete = read_onet(z, "Education.xlsx")

    scale = skills[skills["Scale ID"] == skill_scale].copy()

    if "Recommend Suppress" in scale.columns:
        n_supp = (scale["Recommend Suppress"] == "Y").sum()
        log(f"[onet] skill ratings flagged 'Recommend Suppress' (set to NaN): {n_supp}")
        scale.loc[scale["Recommend Suppress"] == "Y", "Data Value"] = np.nan
    scale["soc_code"] = scale["O*NET-SOC Code"].map(onet_to_soc)

    # Column name per skill:
    # "Critical Thinking" -> "skill_critical_thinking"
    scale["col"] = "skill_" + scale["Element Name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

    skill_wide = scale.pivot_table(index="soc_code", columns="col", values="Data Value", aggfunc="mean")

    # Job zone (2-5): average over sub-types, which is why some values are fractional
    zones["soc_code"] = zones["O*NET-SOC Code"].map(onet_to_soc)
    zone = zones.groupby("soc_code")["Job Zone"].mean().rename("job_zone")


    # Education (1-12)
    # O*NET gives the PERCENT of workers who say each level (1 = less than high school ... 12 = post-doctoral)
    # is required. Average level = sum(level x percent / 100).
    e = ete[(ete["Element Name"] == "Required Level of Education") & (ete["Scale ID"] == "RL")].copy()
    e["soc_code"] = e["O*NET-SOC Code"].map(onet_to_soc)
    e["weighted"] = e["Category"].astype(float) * e["Data Value"] / 100.0
    edu = (e.groupby(["O*NET-SOC Code", "soc_code"])["weighted"].sum()   # average level per O*NET code
             .groupby("soc_code").mean()                                  # then average sub-types per SOC code
             .rename("education_level"))


    # Put job zone, education and skills side by side (axis=1 = as columns)
    onet = pd.concat([zone, edu, skill_wide], axis=1).reset_index().rename(columns={"index": "soc_code"})
    log(f"[onet] SOC occupations: {len(onet)}; skill columns: {skill_wide.shape[1]} (scale {skill_scale})")
    return onet




def main():
    
    OUT.mkdir(parents=True, exist_ok=True)

    exposure = load_exposure()
    cw = load_crosswalk()
    oews = load_oews(cw)
    onet = load_onet()

    step1 = exposure.merge(oews, on="soc_code", how="inner")
    log(f"[join] exposure ∩ OEWS: {len(step1)} "
        f"(lost {len(exposure) - len(step1)} exposure occupations)")
    lost = sorted(set(exposure["soc_code"]) - set(step1["soc_code"]))
    if lost:
        log(f"[join]   first exposure codes not in OEWS: {lost[:15]}")

    full = step1.merge(onet, on="soc_code", how="inner")
    log(f"[join] ... ∩ O*NET: {len(full)} (lost {len(step1) - len(full)} more)")


    # Outcomes: log change = log(new) - log(old); 0.10 is roughly +10%.
    # If either year is missing, the result is NaN automatically.
    full["d_log_emp_2022_2025"] = np.log(full["emp_2025"]) - np.log(full["emp_2022"])    # main outcome
    full["d_log_emp_2016_2019"] = np.log(full["emp_2019"]) - np.log(full["emp_2016"])    # placebo outcome
    full["d_log_wage_2022_2025"] = np.log(full["wage_2025"]) - np.log(full["wage_2022"]) # wage outcome
    usable = full["d_log_emp_2022_2025"].notna().sum()
    log(f"[final] rows: {len(full)}; with a 2022→2025 employment change: {usable}; "
        f"with a 2016→2019 placebo change: {full['d_log_emp_2016_2019'].notna().sum()}")

    full.to_csv(OUT / "occupations.csv", index=False)       # index=False: no extra row-number column
    (OUT / "matching_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nSaved {OUT/'occupations.csv'} and {OUT/'matching_report.txt'}")

if __name__ == "__main__":
    main()