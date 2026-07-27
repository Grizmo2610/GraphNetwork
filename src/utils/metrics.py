import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, precision_recall_curve


def ks_stat(y_true, y_prob):
    order = np.argsort(y_prob)
    y = y_true[order]
    pos_cum = np.cumsum(y) / y.sum()
    neg_cum = np.cumsum(1 - y) / (len(y) - y.sum())
    return np.max(np.abs(pos_cum - neg_cum))


def lift_at_k(y_true, y_prob, k=0.1):
    n = len(y_true)
    topk = max(1, int(n * k))
    order = np.argsort(-y_prob)[:topk]
    return y_true[order].mean() / y_true.mean()


def best_f1(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return np.nanmax(f1)


def all_metrics(y_true, y_prob):
    return dict(auc=roc_auc_score(y_true, y_prob), f1=best_f1(y_true, y_prob),
                ks=ks_stat(y_true, y_prob), lift10=lift_at_k(y_true, y_prob, 0.1))


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y_true, prob_a, prob_b):
    order = np.argsort(-y_true)
    y = y_true[order]
    preds = np.vstack([prob_a[order], prob_b[order]])
    m = int(y.sum())
    n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = 2
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    l = np.array([1.0, -1.0])
    var = l @ cov @ l
    z = (l @ aucs) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], p


def bootstrap_uplift_ci(y_true, prob_base, prob_graph, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.sum() in (0, n):
            diffs[b] = diffs[b - 1] if b > 0 else 0.0
            continue
        diffs[b] = metric_fn(yb, prob_graph[idx]) - metric_fn(yb, prob_base[idx])
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
