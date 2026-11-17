# stats19-ksi-severity

Code, results and dashboard for a group MSc dissertation (UWS, MSc IT with Data Analytics) on predicting the recorded severity of road casualties in Great Britain. The target is killed or seriously injured (KSI) versus slight, using STATS19 data for 2023 to 2025.

Three models are compared: logistic regression, random forest and histogram gradient boosting. Models are developed on 2023 and 2024 data with collision-grouped cross-validation. The 2025 data is kept sealed and used once as the test set.

## Dashboard

Interactive results: https://elvisg-eg.github.io/stats19-ksi-severity/dashboard/stats19_dashboard.html

## Repository layout

- notebooks/  Nine Colab notebooks (01 to 09) and two helper files, stats19_lib.py and stats19_eval.py
- outputs/    Result tables, model records and linkage logs written by the notebooks
- dashboard/  Dashboard source (build_dashboard.py, template.html) and the built stats19_dashboard.html

## How to run

1. Run the notebooks in numerical order, 01 to 09. They were written for Google Colab with files kept in Google Drive. Upload stats19_lib.py and stats19_eval.py to the same Drive folder as the notebooks.
2. Each notebook reads the outputs of the earlier ones, so do not skip steps.
3. The permutation importance in notebook 09 uses three shuffles per field, a departure from the plan that is recorded in outputs/models/final/departures_log.json.

## Rebuild the dashboard

    pip install pandas plotly
    python dashboard/build_dashboard.py

This reads the files in outputs/ and writes dashboard/stats19_dashboard.html.

## Data

STATS19 road safety data from the Department for Transport, published under the Open Government Licence v3.0. The raw data is not stored here. Notebook 01 downloads and links it.

## Files not included

Some large files were left out because of GitHub size limits: the score arrays in outputs/models/search and outputs/models/sensitivity, and the subgroup bootstrap draws for 2025. Notebooks 04, 07 and 08 regenerate them.

## Authors

 Mawuse Darrah, Michael Brown, Melissa Aseye Akpedonu, Elvis Gyamfi.
