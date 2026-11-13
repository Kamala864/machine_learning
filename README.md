# Is AI Changing US Employment?

CS 7830 Machine Learning (Fall 2026), Wright State University. Kamala Joshi and Diwakar Dahal.

Since ChatGPT launched in November 2022, have occupations that use AI the most changed differently from the rest? We compare employment and wage changes from May 2022 to May 2025 across 756 US occupations. We use May 2016 to May 2019 as a pre-ChatGPT placebo.

- **Regression:** Does AI exposure predict employment and wage change, controlling for wage, education and skills?
- **Classification:** Did the occupation grow or shrink?
- **Clustering:** Which kinds of jobs exist in the AI era, based on exposure, wage, education and skills?

## Data

Each row is one occupation (2018 SOC code), not one person.

| Source | Provides |
|---|---|
| Anthropic Economic Index | AI exposure |
| BLS OEWS | Employment and median wage (2016, 2019, 2022, 2025) |
| O*NET | Job zone, education, 35 skills |

## Structure

```
data/raw/          downloaded source files
data/processed/    merged table (occupations.csv)
src/               build_dataset.py
src/analysis/      analysis scripts (eda.py, ...)
results/           figures and tables
reports/           final report
```

## EDA (`src/analysis/eda.py`)

The EDA checks the data before modelling and makes the choices every model depends on. It saves 26 charts to `results/figures/01_*.png`, each titled with its finding and a short note on how to read it. Requires `pandas`, `numpy` and `matplotlib`.

| Step | What it does | Why it matters |
|---|---|---|
| 1. Load and prepare | Loads the data, adds job family, fills top-coded 2022 wages, flags "All Other" jobs and extreme swings | BLS leaves the highest wages blank. Dropping them would remove the top earners. Flags allow robustness checks later. |
| 2. Exposure | Histograms, average by family, most and least exposed jobs | Exposure is the main predictor. Its many zeros may need a yes/no flag alongside the number. |
| 3. Employment and wages | Histograms, raw and log | Both are lopsided, so logs stop a few huge jobs from driving the models |
| 4. Outcomes | Distribution of employment and wage change; biggest gainers and losers; swings by job size | Shows what the models predict. Small jobs swing most, so results should be weighted by employment. |
| 5. Exposure vs change | Scatter plots and group tables for both periods, on the same occupations | First look at the research question. The placebo period shows whether a pattern is new since ChatGPT. |
| 6. Change by family | Box plot by family; exposure vs wage change | Shows non-AI causes (COVID recovery, tech layoffs) that robustness checks must account for |
| 7. Education, job zone, correlations | Distributions, correlation heatmap, VIF | Finds overlapping controls (education and job zone, r = 0.93) that would confuse the regression |
| 8. Skills and PCA | Skill correlations; compresses 35 skills into a few components | Skills overlap heavily. Fewer components avoid overfitting and give features for regression and clustering. |
| 9. Missing values | Counts blanks, checks who is missing, counts rows kept per feature set | Dropping rows that aren't missing at random biases the results. Missing placebo data is a limitation. |
| 10. Scaling | Compares feature contributions to distance before and after standardizing | Without scaling, clusters would just be pay groups |

Key findings:
- 54% of occupations have zero AI exposure. Exposure is highest in Computer & math, Sales, Legal and Office & admin.
- On the same 700 occupations, the most exposed group (>0.3) went from +0.7% growth (2016–19) to −3.2% (2022–25). Zero-exposure jobs barely changed.
- Education and job zone overlap heavily (r = 0.93), so we keep job zone, which has no missing values.
- 4 skill components keep 82.5% of the variance.
- Wage dominates unscaled distances, so features are logged and standardized before modelling.

## Run

```
python src/build_dataset.py
python src/analysis/eda.py
```