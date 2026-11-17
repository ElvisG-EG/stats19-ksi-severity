"""Shared code for the STATS19 severity study (notebooks 03 onward).
Every learned step (medians, rare-category pooling, scaling, spline knots, oversampling)
is fitted on the training fold only, inside fit_predict()."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from imblearn.over_sampling import RandomOverSampler

RARE_MIN_FREQ = 100          # categories with fewer training-fold casualties are pooled into one level
RF_TREES = 100               # fixed; the number of trees is not tuned (3.7.2)
KINDS = ['lr', 'rf', 'hgb']
TREATMENTS = ['untreated', 'weight', 'oversample']

def load_dev(project):
    project = Path(project)
    summ = json.loads((project / 'outputs' / 'features' / 'preprocessing_summary.json').read_text())
    p = project / 'data' / 'model_inputs'
    f = p / 'features_dev_2023_2024.parquet'
    df = pd.read_parquet(f) if f.exists() else pd.read_pickle(p / 'features_dev_2023_2024.pkl')
    return df, summ['numeric'], summ['categorical']

def make_preprocessor(kind, num, cat):
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=RARE_MIN_FREQ,
                        sparse_output=(kind == 'lr'), dtype=np.float32)
    if kind == 'lr':
        age = [c for c in ('age_of_casualty',) if c in num]
        other = [c for c in num if c not in age]
        tf = []
        if age:
            tf.append(('age_spline', Pipeline([('imp', SimpleImputer(strategy='median')),
                       ('spl', SplineTransformer(n_knots=6, degree=3, knots='quantile', include_bias=False))]), age))
            tf.append(('age_missing', MissingIndicator(features='all'), age))
        if other:
            tf.append(('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                                        ('sc', StandardScaler())]), other))
    elif kind == 'rf':
        tf = [('num', SimpleImputer(strategy='median', add_indicator=True), num)] if num else []
    else:  # hgb handles missing numerics natively
        tf = [('num', 'passthrough', num)] if num else []
    tf.append(('cat', ohe, cat))
    return ColumnTransformer(tf, remainder='drop', sparse_threshold=(0.3 if kind == 'lr' else 0.0),
                             verbose_feature_names_out=False)

ANCHOR = {
    'lr': dict(C=0.1),
    'rf': dict(max_features='sqrt', min_samples_leaf=20, min_samples_split=20),
    'hgb': dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0),
}
SPACE = {
    'lr': dict(C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
    'rf': dict(max_features=['sqrt', 0.1, 0.2], min_samples_leaf=[5, 10, 20, 50], min_samples_split=[10, 20, 50]),
    'hgb': dict(learning_rate=[0.03, 0.05, 0.1], max_iter=[100, 200, 300, 500], max_leaf_nodes=[15, 31, 63],
                min_samples_leaf=[20, 50, 100], l2_regularization=[0.0, 1.0, 10.0]),
}

def sample_configs(kind, n, seed):
    """Anchor configuration first, then a seeded random draw without replacement from the grid."""
    import itertools
    keys = list(SPACE[kind])
    grid = [dict(zip(keys, v)) for v in itertools.product(*[SPACE[kind][k] for k in keys])]
    if kind == 'lr':
        idx = np.unique(np.linspace(0, len(grid) - 1, num=min(n, len(grid))).round().astype(int))
        return [grid[i] for i in idx]
    anchor = ANCHOR[kind]
    rest = [g for g in grid if g != anchor]
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(rest), size=min(n - 1, len(rest)), replace=False)
    return [anchor] + [rest[i] for i in pick]

def make_model(kind, params, treatment, seed, n_jobs=-1):
    cw = 'balanced' if treatment == 'weight' else None
    if kind == 'lr':
        return LogisticRegression(C=params['C'], solver='lbfgs', max_iter=300, class_weight=cw, random_state=seed)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=RF_TREES, max_features=params['max_features'],
                                      min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'],
                                      class_weight=cw, n_jobs=n_jobs, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=params['max_iter'], learning_rate=params['learning_rate'],
                                          max_leaf_nodes=params['max_leaf_nodes'], min_samples_leaf=params['min_samples_leaf'],
                                          l2_regularization=params['l2_regularization'], early_stopping=False,
                                          class_weight=cw, random_state=seed)

def _nbytes(a):
    return (a.data.nbytes + a.indices.nbytes + a.indptr.nbytes) if hasattr(a, 'indptr') else a.nbytes

def fit_predict(kind, params, treatment, Xtr, ytr, Xva, num, cat, seed, n_jobs=-1):
    t0 = time.time()
    pre = make_preprocessor(kind, num, cat)
    Atr = pre.fit_transform(Xtr[num + cat]); Ava = pre.transform(Xva[num + cat])
    if kind != 'lr':
        Atr = Atr.astype(np.float32); Ava = Ava.astype(np.float32)
    n_cols = Atr.shape[1]
    y = np.asarray(ytr)
    if treatment == 'oversample':
        Atr, y = RandomOverSampler(sampling_strategy=1.0, random_state=seed).fit_resample(Atr, y)
    t_prep = time.time() - t0
    model = make_model(kind, params, treatment, seed, n_jobs)
    t1 = time.time(); model.fit(Atr, y); t_fit = time.time() - t1
    t2 = time.time(); p = model.predict_proba(Ava)[:, 1]; t_pred = time.time() - t2
    info = dict(n_cols=int(n_cols), n_train=int(Atr.shape[0]), matrix_mb=round(_nbytes(Atr) / 1e6, 1),
                t_prep=round(t_prep, 2), t_fit=round(t_fit, 2), t_pred=round(t_pred, 2))
    return p, info

def metrics(y, p):
    return dict(ap=average_precision_score(y, p), auc=roc_auc_score(y, p), brier=brier_score_loss(y, p))
