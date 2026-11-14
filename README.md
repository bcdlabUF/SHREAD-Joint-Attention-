# SHREAD: Data and analysis code

Data and code to reproduce the statistics and figures in:

> Newland, J., Aukerman, J. F., Cruz, K., Burgos Villanueva, V., Keil, A., & Scott, L. S. Follow the Leader: Infant Attention, Joint Visual Attention, Coordination, and Reciprocity During Shared Book Reading at 6, 9, and 12 Months.

Brain, Cognition & Development Lab, University of Florida. Contact: Lisa S. Scott.

## Contents

```
SHREAD_paper1/
├── README.md
├── run_all.sh                 runs every step in order, then sorts the outputs
├── data/                      coded data (numbers and codes only; no video, audio, or transcripts)
├── code/                      analysis and figure scripts (Python)
│   └── not_in_this_paper/     scripts for the companion talk-and-gesture paper and exploratory checks; not run
└── outputs/                   results of run_all.sh, as used for the manuscript
    ├── main_paper/
    │   ├── statistics/        printed statistics for the Method and Results, one file per script
    │   ├── figures/           Figures 3–7
    │   └── data_tables/       per-dyad measures behind the Results and figures
    ├── supplementary/
    │   ├── statistics/        printed statistics for Supplementary Materials S3–S5
    │   ├── figures/           Figures S1–S2
    │   └── data_tables/
    └── shared/                intermediate files used by both (e.g., the 30-Hz looking series)
```

Running `run_all.sh` also creates `results/` and `figures/` (the working folders) and rebuilds `outputs/`.

## How to run

Requires Python 3.11 or later and numpy, pandas, scipy, statsmodels, pingouin, scikit-posthocs, scikit-learn, patsy, and matplotlib.

```bash
pip install numpy pandas scipy statsmodels pingouin scikit-posthocs scikit-learn patsy matplotlib
bash run_all.sh
```

The full run takes roughly 10–20 minutes; most of the time goes to the circular-shift chance baselines and the Granger models. Random baselines use fixed seeds, so repeated runs give the same numbers.

## Sample

60 parent–infant dyads: 20 at 6 months, 23 at 9 months, and 17 at 12 months (`data/included_dyads.csv`).

## Where each result comes from

### Main paper

| Manuscript section | Script(s) | Statistics file (`outputs/main_paper/statistics/`) |
|---|---|---|
| Preliminary analyses (sex, book version) | `a1.py` | `a1.txt` |
| Sensitivity power analysis (Data Analysis) | `power_sensitivity.py` | `power_sensitivity.txt` |
| Infant attention to the book, sustained attention, parent attention to the book | `final_stats.py`, `yu_defs.py`, `paper_stats.py` | `final_stats.txt`, `yu_defs.txt`, `paper_stats.txt` |
| Joint visual attention, comparison with chance, share of infant looking in joint attention | `chance_unified.py`, `paper_stats.py` | `chance_unified.txt`, `paper_stats.txt` |
| Coordination (cross-correlation, lag 0 vs. longer lags) | `xc2.py`, `xc3.py`, `xc4.py` | `xc3.txt`, `xc4.txt` |
| Who led joint visual attention | `lead.py`, `chance_unified.py`, `paper_stats.py` | `paper_stats.txt` |
| Lead changes (windowed cross-correlation) | `wxc_all.py` | `wxc_all.txt`, `paper_stats.txt` |
| Who predicted whom (Granger causality) | `granger.py` | `granger.txt`, `paper_stats.txt` |
| Page type | `page_condition.py`, `lead_by_page.py`, `page_condition_stats.py` | `page_condition_stats.txt` |
| Number of readings (Procedure) | `page_rereading.py` | `outputs/supplementary/statistics/page_rereading.txt` |

| Figure | Script | File (`outputs/main_paper/figures/`) |
|---|---|---|
| Figure 3 | `fig_methods_excerpt.py` | `Figure3_coordination_reciprocity_example.png` |
| Figure 4 | `docfig1_ja.py` | `Figure4_attention_joint_attention_by_age.png` |
| Figure 5 | `fig_xcorr_main.py` | `Figure5_coordination.png` |
| Figure 6 | `fig_reciprocity.py` | `Figure6_reciprocity.png` |
| Figure 7 | `fig_page_type_sig.py` | `Figure7_page_type.png` |

### Supplementary Materials

| Section | Script(s) | Statistics file (`outputs/supplementary/statistics/`) |
|---|---|---|
| S2. Interrater reliability | Computed from the two coders' files, which are not included | — |
| S3. Session length and time on task | `a1.py` (session length), `extras.py` (first vs. second half), `page_condition_stats.py` (GEEs with session time), `page_rereading.py` (by reading) | `extras.txt`, `page_rereading.txt`; GEEs in `main_paper/statistics/page_condition_stats.txt` |
| S4. Robustness of age effects (nonparametric tests, pseudo-dyad chance, who led controlling for looking) | `paper_stats.py`, `xc3.py`, `xc4.py`, `final_stats.py` | In the corresponding `main_paper/statistics/` files |
| S5. Sensitivity of Granger causality results | `granger_sens.py`, `granger_stationarity.py` | `granger_sens.txt`, `granger_stationarity.txt` |

| Figure | Script | File (`outputs/supplementary/figures/`) |
|---|---|---|
| Figure S1 | `fig_page_type_age.py` | `FigureS1_page_type_by_age.png` |
| Figure S2 | `fig_page_type_time.py` | `FigureS2_page_type_time_on_task.png` |

Several scripts print both main-paper and supplementary statistics. Their output is filed under `main_paper/statistics/`, and the table above says which supplementary results they contain.

### Shared preparation

`a1.py` (per-dyad measures), `xc.py` (30-Hz looking series, `series.pkl`), `xc2.py` (lagged cross-correlations), and `jadef.py` (joint-attention definition: bouts > 500 ms with gaps < 300 ms bridged; Yu et al., 2019). `final_stats.py` also reads event files made by `booktalk.py`, `kinds.py`, `infant.py`, `maint.py`, and `breaker.py`; those scripts code parent talk and gesture events for the companion paper (Aukerman et al., submitted) and are run only as inputs. `docfigs.py` holds shared figure styling and reads `page_dyad.csv` from `page_collapsed.py`.

## Data files

All times are in milliseconds from the start of the coded session.

**`look_spans.csv`**: one row per coded span (`dyad_id`, `age_group`, `stream`, `onset_ms`, `offset_ms`, `value`).

| `stream` | Meaning |
|---|---|
| `infant_book`, `parent_book` | Infant / parent looking at the book |
| `infant_partner`, `parent_partner` | Looks to the partner's face (coded; not analyzed in this paper) |
| `parent_gesture` | Parent gesture onsets (companion paper) |
| `condition` | The page the book was open to (individual, category, none, break) |

**`talk_timing.csv`**: one row per parent extratextual utterance (timing and talk codes; used in this paper only as an input to shared preparation).

**`covariates.csv`**: one row per dyad: age group, book version, infant and parent sex, session length and time on each page type, and summary measures.

**`included_dyads.csv`**: the 60 included dyad IDs.

## Statistical approach (summary)

- The dyad is the unit of analysis; frame- and event-level measures are summarized within each dyad first.
- Age effects: one-way ANOVAs (partial η²) with Tukey HSD; Kruskal–Wallis tests with Bonferroni-corrected Dunn comparisons where residuals were not normal.
- Chance: circular shifts of at least 20 s (each partner's looking preserved, real pairing broken); observed vs. chance with paired t-tests, Holm-corrected across ages.
- Granger causality: log-likelihood gain per 1,000 200-ms bins from adding the other partner's past 2 s to a logistic autoregression; excess over circular-shift chance; 2 (direction) × 3 (age) mixed ANOVA.
- Page type: 3 (page type) × 3 (age) mixed ANOVAs with Greenhouse–Geisser correction and Friedman checks; GEEs on continuous stretches on one page type (≥ 3 s) with and without session time.
- Tests are two-tailed, α = .05.
