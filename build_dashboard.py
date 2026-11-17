"""Build the self-contained dashboard (stats19_dashboard.html).

Run from the repository root:
    pip install pandas plotly
    python dashboard/build_dashboard.py
Reads the result tables in outputs/ (features/, models/final/, results/) and
writes dashboard/stats19_dashboard.html with the data and Plotly embedded.
"""
import pandas as pd, json, os, plotly
HERE=os.path.dirname(os.path.abspath(__file__))
O=os.path.join(HERE,'..','outputs')+os.sep
def rd(p): return pd.read_csv(O+p)
def recs(df): return json.loads(df.to_json(orient='records'))
fd=rd('features/field_dictionary.csv'); pa=rd('features/predictor_availability_by_year.csv')
prim=set(fd[fd.status.str.startswith('primary')].field)
pa_p=pa[pa.field.isin(prim)].copy(); pa_p['mx']=pa_p[['unknown_2023','unknown_2024','unknown_2025']].max(axis=1)
pa_p=pa_p.sort_values('mx',ascending=False).head(12)
sens=rd('results/sensitivity_2025_measures.csv')
D=dict(
 fold=recs(rd('features/fold_summary.csv')),
 avail=recs(pa_p), status_counts=fd.status.value_counts().to_dict(),
 prev=recs(rd('results/ksi_prevalence_by_year_and_group.csv')),
 rank=recs(rd('models/final/model_ranking_dev.csv')),
 foldf1=recs(rd('models/final/fold_macro_f1_dev.csv')),
 thr=recs(rd('models/final/thresholds_dev.csv')),
 dev25=recs(rd('results/development_vs_2025.csv')),
 score=recs(rd('results/results_2025_score_measures.csv')),
 dec=recs(rd('results/results_2025_decisions.csv')),
 paired=recs(rd('results/paired_contrasts_2025.csv')),
 rel=recs(rd('results/reliability_bins_2025.csv')),
 tvu=recs(rd('results/treated_vs_untreated_share_better_2025.csv')),
 rstab=recs(rd('results/ranking_stability_2025.csv')),
 sgo=recs(rd('results/subgroups_2025_overview.csv')),
 sgs=recs(rd('results/subgroups_2025_score_measures.csv')),
 sgd=recs(rd('results/subgroups_2025_decision_measures.csv')),
 decomp=recs(rd('results/ksi_change_decomposition_2025_vs_2023_24.csv')),
 radj=recs(rd('results/recorded_vs_adjusted_ksi_annual.csv')),
 sens=recs(sens), sgap=recs(rd('results/sensitivity_2025_system_calibration_gap.csv')),
 temporal=recs(rd('results/temporal_2023_to_2024_vs_2025_test.csv')),
 imp=recs(rd('results/permutation_importance_dev_summary.csv')),
 imp25=recs(rd('results/permutation_importance_2025_vs_dev.csv')),
 departures=json.load(open(O+'models/final/departures_log.json')),
 freeze=json.load(open(O+'models/final/freeze_record.json')),
)
js=open(os.path.dirname(plotly.__file__)+'/package_data/plotly.min.js').read()
tpl=open(os.path.join(HERE,'template.html'),encoding='utf-8').read()
html=tpl.replace('/*PLOTLY*/',js).replace('/*DATA*/','const D='+json.dumps(D,separators=(',',':'))+';')
open(os.path.join(HERE,'stats19_dashboard.html'),'w',encoding='utf-8').write(html)
print(len(html)/1e6,'MB')
