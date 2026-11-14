#!/usr/bin/env bash
# Reproduce the analyses and figures for:
#   "Follow the Leader: Infant Attention, Joint Visual Attention, Coordination, and Reciprocity
#    During Shared Book Reading at 6, 9, and 12 Months"
# Run from this folder:  bash run_all.sh
# Working files go to results/. At the end, outputs are sorted into
#   outputs/main_paper/      (Method and Results, Figures 3-7)
#   outputs/supplementary/   (Supplementary Materials S3-S5, Figures S1-S2)
#   outputs/shared/          (intermediate data files used by both)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
cd results
export PYTHONPATH=../code MPLBACKEND=Agg
run () { echo ">>> $*"; python ../code/"$@" > "log_${1%.py}.txt" 2>"err_${1%.py}.txt"; }

# ---------------------------------------------------------------------------------------------
# 0. SHARED PREPARATION (needed by the main paper and the supplement)
# ---------------------------------------------------------------------------------------------
run a1.py              # per-dyad measures; sex, book version, session length checks -> cov_derived.csv
run xc.py              # 30-Hz binary looking series for each dyad -> series.pkl
run xc2.py             # lagged cross-correlations (-10 to +10 s) -> xc_dyad.csv, xc_long.csv
# Event files required by final_stats.py (which writes final_age.csv, the per-dyad age table).
# These scripts code parent talk and gesture events; those results are reported in the
# companion paper (Aukerman et al., submitted), not in this one.
run booktalk.py        # talk_cat.csv
run kinds.py           # kinds_events.csv
run infant.py          # infant_events.csv
run maint.py           # maint_events.csv
run breaker.py         # breaker_events.csv
run page_collapsed.py  # page_dyad.csv (read by the shared figure helpers)

# ---------------------------------------------------------------------------------------------
# 1. MAIN PAPER
# ---------------------------------------------------------------------------------------------
# Q1. Infant attention, sustained attention, parent looking, joint visual attention, chance
run yu_defs.py         # joint-attention and sustained-attention definitions (Yu et al., 2019)
run final_stats.py     # age ANOVAs -> final_age.csv
run chance_unified.py  # circular-shift chance for joint attention and who led
# Q2. Coordination and reciprocity
run xc3.py             # coordination: age ANOVA, circular-shift chance (also S4 pseudo-dyad chance)
run xc4.py             # lag-0 vs longer-lag coordination (also S4 checks)
run lead.py            # who led each joint-attention bout
run wxc_all.py         # windowed cross-correlation: coupling and lead changes
run granger.py         # Granger causality, parent <-> infant book looking
run paper_stats.py     # summary statistics reported in the Results (also S4 nonparametric checks)
# Q3. Page type
run page_condition.py        # per dyad x page type measures; per-stretch data with session time
run lead_by_page.py          # who led by page type, excess over circular-shift chance
run page_condition_stats.py  # 3 x 3 mixed ANOVAs, Holm pairs, GEEs with session time
# Method
run power_sensitivity.py     # sensitivity power analysis (Data Analysis)
# Figures
run fig_methods_excerpt.py   # Figure 3
run docfig1_ja.py            # Figure 4
run fig_xcorr_main.py        # Figure 5
run fig_reciprocity.py       # Figure 6
run fig_page_type_sig.py     # Figure 7

# ---------------------------------------------------------------------------------------------
# 2. SUPPLEMENTARY MATERIALS
# ---------------------------------------------------------------------------------------------
# S2 (interrater reliability) was computed from the two coders' files, which are not included here.
run extras.py                # S3: first vs second half of reading time
run page_rereading.py        # S3: re-reading (also the number of readings in the Method)
run fig_page_type_time.py    # S3: Figure S2 (page type and time on task; GEEs from page_condition_stats.py)
run granger_sens.py          # S5: Granger bin size and history length
run granger_stationarity.py  # S5: ADF/KPSS; Granger with session-time and page-state controls
run fig_page_type_age.py     # Figure S1 (attention by page type and age group)

cd ..
python code/organize_outputs.py
echo "Done. See outputs/main_paper and outputs/supplementary."
