"""Evaluation helpers for the STATS19 severity study (notebooks 06 onward).
Everything here is label-agnostic code: it only runs once the sealed 2025 labels are opened in notebook 06."""
import numpy as np
from scipy.special import expit, logit

CLIP = 1e-6                      # probabilities are clipped to [CLIP, 1 - CLIP] only for the logit calculation

def clip_logit(p):
    return logit(np.clip(p, CLIP, 1 - CLIP))

def irls_logistic(x, y, w=None, fit_slope=True, start=(0.0, 1.0), iters=60, tol=1e-9):
    """Weighted logistic regression of y on x. fit_slope=False fixes the slope at 1 (intercept only).
    Returns (intercept, slope)."""
    w = np.ones(len(y)) if w is None else w
    a, b = start if fit_slope else (start[0], 1.0)
    for _ in range(iters):
        mu = expit(a + b * x)
        s = w * mu * (1.0 - mu)
        r = w * (y - mu)
        if fit_slope:
            sx = (s * x).sum()
            H = np.array([[s.sum(), sx], [sx, (s * x * x).sum()]])
            step = np.linalg.solve(H, np.array([r.sum(), (r * x).sum()]))
            a += step[0]; b += step[1]
        else:
            step = np.array([r.sum() / s.sum()]); a += step[0]
        if np.max(np.abs(step)) < tol:
            break
    return float(a), float(b)

class RankPrep:
    """Pre-sorts scores once so weighted average precision and ROC-AUC are cheap inside a bootstrap loop.
    Tied scores form one threshold step, as in scikit-learn."""
    def __init__(self, p, y):
        order = np.argsort(-p, kind='mergesort')
        ps = p[order]
        new = np.r_[True, ps[1:] != ps[:-1]]
        self.gid = np.cumsum(new) - 1
        self.ng = int(self.gid[-1]) + 1
        self.order = order
        self.ys = y[order].astype(float)
    def ap_auc(self, w):
        ws = w[self.order]
        pos = np.bincount(self.gid, weights=ws * self.ys, minlength=self.ng)
        neg = np.bincount(self.gid, weights=ws * (1.0 - self.ys), minlength=self.ng)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        P, N = tp[-1], fp[-1]
        den = tp + fp
        prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
        rec, fpr = tp / P, fp / N
        ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        auc = float(np.sum(np.diff(np.r_[0.0, fpr]) * (np.r_[0.0, rec[:-1]] + rec) / 2.0))
        return ap, auc

def decision_rates(c):
    """c[..., 0:4] = tp, fp, fn, tn. Returns the Table 3.3 decision measures."""
    c = np.asarray(c, dtype=float)
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    n = tp + fp + fn + tn
    rec = tp / (tp + fn); spec = tn / (tn + fp)
    prec = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    f_pos = 2 * tp / (2 * tp + fp + fn); f_neg = 2 * tn / (2 * tn + fn + fp)
    return dict(recall=rec, precision=prec, specificity=spec, balanced_accuracy=(rec + spec) / 2,
                macro_f1=(f_pos + f_neg) / 2, flagged_share=(tp + fp) / n)

def score_metrics(prep, x_logit, p, y, w, start=(0.0, 1.0)):
    """Average precision, ROC-AUC, Brier score, calibration intercept (slope fixed at 1) and calibration slope."""
    ap, auc = prep.ap_auc(w)
    brier = float(np.sum(w * (p - y) ** 2) / np.sum(w))
    a_int, _ = irls_logistic(x_logit, y, w, fit_slope=False)
    a2, b2 = irls_logistic(x_logit, y, w, fit_slope=True, start=start)
    return dict(ap=ap, auc=auc, brier=brier, cal_intercept=a_int, cal_slope=b2), (a2, b2)

def collision_counts(coll_codes, n_coll, flag, y):
    """Per-collision tp, fp, fn, tn for one decision rule (flag and y are 0/1 arrays)."""
    f = flag.astype(float); yy = y.astype(float)
    tp = np.bincount(coll_codes, weights=f * yy, minlength=n_coll)
    fp = np.bincount(coll_codes, weights=f * (1 - yy), minlength=n_coll)
    fn = np.bincount(coll_codes, weights=(1 - f) * yy, minlength=n_coll)
    tn = np.bincount(coll_codes, weights=(1 - f) * (1 - yy), minlength=n_coll)
    return np.stack([tp, fp, fn, tn], axis=1)
