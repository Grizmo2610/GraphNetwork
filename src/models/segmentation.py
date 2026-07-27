import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split


from src.services.simulation.sim_data import SCALE_CONFIG, generate_dataset
from src.services.processing.graph_features import build_graph, compute_features, _sparse_adj
from src.utils.metrics import delong_test
from .supply_chain import make_model, HAS_LGBM

DEMO_COLS = ["debt_amount", "overdue_days", "income", "age", "employment_years", "application_amount"]
GRAPH_EXTRA_COLS = ["degree", "pagerank", "clustering", "neighbor_mean_debt"]


def ari_bootstrap_ci(true, labels_base, labels_graph, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    n = len(true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = adjusted_rand_score(true[idx], labels_graph[idx]) - adjusted_rand_score(true[idx], labels_base[idx])
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def run_scale(name, cfg):
    customers, edges = generate_dataset(**cfg)
    G = build_graph(customers, edges)
    feats = compute_features(G, customers, edges)
    df = customers.merge(feats, on="id")
    n = len(df)
    n_comm = max(10, cfg["n_customers"] // 150)

    Xb = StandardScaler().fit_transform(df[DEMO_COLS])
    Xg = StandardScaler().fit_transform(df[DEMO_COLS + GRAPH_EXTRA_COLS])

    km_b = KMeans(n_clusters=n_comm, n_init=8, random_state=42).fit(Xb)
    km_g = KMeans(n_clusters=n_comm, n_init=8, random_state=42).fit(Xg)

    true = df["_true_segment"].to_numpy()
    ari_b = adjusted_rand_score(true, km_b.labels_)
    ari_g = adjusted_rand_score(true, km_g.labels_)
    ari_uplift, ari_lo, ari_hi = ari_bootstrap_ci(true, km_b.labels_, km_g.labels_)
    sample_size = min(3000, n)
    sil_b = silhouette_score(Xb, km_b.labels_, sample_size=sample_size, random_state=42)
    sil_g = silhouette_score(Xg, km_g.labels_, sample_size=sample_size, random_state=42)

    rng2 = np.random.default_rng(123)
    segment_propensity = rng2.normal(0, 1, n_comm)[true]

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    logit = (0.5 * z(df["income"].to_numpy()) - 0.3 * z(df["age"].to_numpy())
             + 1.2 * segment_propensity + rng2.normal(0, 0.7, n))
    adopted = rng2.binomial(1, 1 / (1 + np.exp(-logit)))
    adj = _sparse_adj(df, edges)
    deg_safe = np.maximum(df["degree"].to_numpy(), 1)
    neighbor_adopt_rate = (adj @ adopted) / deg_safe

    Xb2 = df[["income", "age"]].copy()
    Xg2 = Xb2.copy()
    Xg2["neighbor_adopt_rate"] = neighbor_adopt_rate
    Xg2["degree"] = df["degree"]
    Xg2["pagerank"] = df["pagerank"]

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.3, stratify=adopted, random_state=42)
    model_b = make_model().fit(Xb2.iloc[train_idx], adopted[train_idx])
    model_g = make_model().fit(Xg2.iloc[train_idx], adopted[train_idx])
    prob_b = model_b.predict_proba(Xb2.iloc[test_idx])[:, 1]
    prob_g = model_g.predict_proba(Xg2.iloc[test_idx])[:, 1]
    y_test = adopted[test_idx]
    auc_b, auc_g, delong_p = delong_test(y_test, prob_b, prob_g)

    return dict(
        name=name, n=n, n_segments_true=int(n_comm),
        ari_base=float(ari_b), ari_graph=float(ari_g), ari_uplift=float(ari_uplift),
        ari_ci=[float(ari_lo), float(ari_hi)], pass_ari=bool(ari_lo > 0),
        sil_base=float(sil_b), sil_graph=float(sil_g),
        adopt_auc_base=float(auc_b), adopt_auc_graph=float(auc_g),
        adopt_auc_uplift=float(auc_g - auc_b), adopt_delong_p=float(delong_p),
        pass_adopt=bool((auc_g - auc_b) > 0 and delong_p < 0.05),
    )


def run():
    print(f"[Customer Segmentation] model backend: {'LightGBM' if HAS_LGBM else 'sklearn HistGradientBoostingClassifier'}")
    scales = []
    for name, cfg in SCALE_CONFIG.items():
        r = run_scale(name, cfg)
        scales.append(r)
        print(f"{name}: ARI base={r['ari_base']:.4f} graph={r['ari_graph']:.4f} uplift={r['ari_uplift']:+.4f} "
              f"CI[{r['ari_ci'][0]:+.4f},{r['ari_ci'][1]:+.4f}] -> {'PASS' if r['pass_ari'] else 'FAIL'}  "
              f"silhouette base={r['sil_base']:.4f} graph={r['sil_graph']:.4f}  "
              f"adoption AUC base={r['adopt_auc_base']:.4f} graph={r['adopt_auc_graph']:.4f} "
              f"p={r['adopt_delong_p']:.4g} -> {'PASS' if r['pass_adopt'] else 'FAIL'}")
    return dict(problem="Customer Segmentation — phân khúc theo cấu trúc quan hệ + dự đoán nhu cầu sản phẩm", scales=scales)


if __name__ == "__main__":
    report = run()
    with open("metrics_segmentation.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved metrics_segmentation.json")
