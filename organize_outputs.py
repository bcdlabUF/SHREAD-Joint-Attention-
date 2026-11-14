"""Sort pipeline outputs (results/, figures/) into outputs/main_paper, outputs/supplementary and outputs/shared.
Statistics logs are copied as <script>.txt; figures are renamed to their manuscript numbers."""
import os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
O = 'outputs'
shutil.rmtree(O, ignore_errors=True)

MAIN_LOGS = ['a1', 'yu_defs', 'final_stats', 'chance_unified', 'xc3', 'xc4', 'lead', 'wxc_all', 'granger',
             'paper_stats', 'page_condition', 'lead_by_page', 'page_condition_stats', 'power_sensitivity']
SUPP_LOGS = ['extras', 'page_rereading', 'granger_sens', 'granger_stationarity']
MAIN_FIGS = {'fig_methods_excerpt.png': 'Figure3_coordination_reciprocity_example.png',
             'fig_attention_ja_by_age.png': 'Figure4_attention_joint_attention_by_age.png',
             'fig_xcorr_main.png': 'Figure5_coordination.png',
             'Fig6_reciprocity.png': 'Figure6_reciprocity.png',
             'Fig7_page_type.png': 'Figure7_page_type.png'}
SUPP_FIGS = {'FigS1_page_type_by_age.png': 'FigureS1_page_type_by_age.png',
             'FigS2_page_type_time.png': 'FigureS2_page_type_time_on_task.png'}
MAIN_TABLES = ['cov_derived.csv', 'final_age.csv', 'chance_unified.csv', 'xc_dyad_book.csv', 'lead_dyad.csv',
               'wxc_dyad.csv', 'granger_dyad.csv', 'page_condition_dyad.csv', 'lead_by_page.csv', 'page_condition_table.csv']
SUPP_TABLES = ['page_condition_spans.csv', 'page_rereading_dyad.csv', 'page_rereading_spans.csv',
               'granger_detrended.csv', 'granger_stationarity_tests.csv']
SHARED = ['series.pkl', 'xc_dyad.csv', 'xc_long.csv', 'lead_bouts.csv', 'yu_defs_dyad.csv', 'page_dyad.csv']

def cp(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
    else:
        print('  missing:', src)

for part, logs, figs, tabs in [('main_paper', MAIN_LOGS, MAIN_FIGS, MAIN_TABLES), ('supplementary', SUPP_LOGS, SUPP_FIGS, SUPP_TABLES)]:
    for l in logs: cp(f'results/log_{l}.txt', f'{O}/{part}/statistics/{l}.txt')
    for a, b in figs.items(): cp(f'figures/{a}', f'{O}/{part}/figures/{b}')
    for t in tabs: cp(f'results/{t}', f'{O}/{part}/data_tables/{t}')
for s in SHARED: cp(f'results/{s}', f'{O}/shared/{s}')
print('outputs sorted into', O)
