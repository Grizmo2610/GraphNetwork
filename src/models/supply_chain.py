import time
import json
import tracemalloc
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split

from src.services.simulation.sim_data import SCALE_CONFIG, generate_dataset
from src.services.processing.graph_features import build_graph, compute_features, GRAPH_FEATURE_COLS
from src.utils.metrics import all_metrics, delong_test, bootstrap_uplift_ci, best_f1, ks_stat, lift_at_k

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True

    def make_model():
        return LGBMClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, verbosity=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False

    def make_model():
        return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=250,
                                               categorical_features="from_dtype")

try:
    import shap
    HAS_SHAP = True

    def rank_features(model, X, y):
        sv = shap.TreeExplainer(model).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
except ImportError:
    from sklearn.inspection import permutation_importance
    HAS_SHAP = False

    def rank_features(model, X, y):
        r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=8, random_state=42)
        return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


BASE_COLS = ["debt_amount", "overdue_days", "income", "age", "employment_years",
             "application_amount", "n_phones", "phone_active_ratio", "address_active",
             "n_alt_contacts", "n_alt_contacts_active", "still_employed"]


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def mem_call(fn):
    tracemalloc.start()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1e6


def run_scale(name, cfg):
    print(f"\n=== {name} (n_customers={cfg['n_customers']}) ===")
    t0 = time.perf_counter()
    customers, edges = generate_dataset(**cfg)
    t_gen = time.perf_counter() - t0

    (feats, t_feat), mem_feat = mem_call(lambda: time_call(
        lambda: compute_features(build_graph(customers, edges), customers, edges)))
    graph_df = customers.merge(feats, on="id")

    train_idx, test_idx = train_test_split(np.arange(len(graph_df)), test_size=0.3,
                                            stratify=graph_df["default"], random_state=42)
    y = graph_df["default"].to_numpy()

    Xb = graph_df[BASE_COLS]
    Xg = graph_df[BASE_COLS + GRAPH_FEATURE_COLS]

    t0 = time.perf_counter()
    model_b = make_model().fit(Xb.iloc[train_idx], y[train_idx])
    t_train_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    model_g = make_model().fit(Xg.iloc[train_idx], y[train_idx])
    t_train_g = time.perf_counter() - t0

    prob_b = model_b.predict_proba(Xb.iloc[test_idx])[:, 1]
    prob_g = model_g.predict_proba(Xg.iloc[test_idx])[:, 1]
    y_test = y[test_idx]

    mb, mg = all_metrics(y_test, prob_b), all_metrics(y_test, prob_g)
    auc_b, auc_g, delong_p = delong_test(y_test, prob_b, prob_g)

    f1_up, f1_lo, f1_hi = bootstrap_uplift_ci(y_test, prob_b, prob_g, best_f1)
    ks_up, ks_lo, ks_hi = bootstrap_uplift_ci(y_test, prob_b, prob_g, ks_stat)
    lift_up, lift_lo, lift_hi = bootstrap_uplift_ci(
        y_test, prob_b, prob_g, lambda yt, yp: lift_at_k(yt, yp, 0.1))

    importance = rank_features(model_g, Xg.iloc[test_idx], y_test)
    top5 = importance.head(5)
    graph_in_top5 = any(c in GRAPH_FEATURE_COLS for c in top5.index)

    pass_auc = (mg["auc"] - mb["auc"] > 0) and (delong_p < 0.05)
    pass_f1 = f1_lo > 0
    pass_ks = ks_lo > 0
    pass_lift = lift_lo > 0
    accept = pass_auc and pass_f1 and pass_ks and pass_lift and graph_in_top5

    print(f"AUC   base={mb['auc']:.4f} graph={mg['auc']:.4f} uplift={mg['auc']-mb['auc']:+.4f} "
          f"DeLong p={delong_p:.4g} -> {'PASS' if pass_auc else 'FAIL'}")
    print(f"F1    base={mb['f1']:.4f} graph={mg['f1']:.4f} uplift={f1_up:+.4f} "
          f"95%CI[{f1_lo:+.4f},{f1_hi:+.4f}] -> {'PASS' if pass_f1 else 'FAIL'}")
    print(f"KS    base={mb['ks']:.4f} graph={mg['ks']:.4f} uplift={ks_up:+.4f} "
          f"95%CI[{ks_lo:+.4f},{ks_hi:+.4f}] -> {'PASS' if pass_ks else 'FAIL'}")
    print(f"Lift@10 base={mb['lift10']:.4f} graph={mg['lift10']:.4f} uplift={lift_up:+.4f} "
          f"95%CI[{lift_lo:+.4f},{lift_hi:+.4f}] -> {'PASS' if pass_lift else 'FAIL'}")
    print(f"Top-5 features ({'SHAP' if HAS_SHAP else 'permutation'}): "
          + ", ".join(top5.index) + f"  -> graph feature in top5: {graph_in_top5}")
    print(f"Decision: {'ACCEPT graph model' if accept else 'REJECT graph model'}")

    metrics_record = dict(
        name=name, n=cfg["n_customers"], n_edges=len(edges),
        auc_base=mb["auc"], auc_graph=mg["auc"], auc_uplift=mg["auc"] - mb["auc"], delong_p=delong_p,
        f1_base=mb["f1"], f1_graph=mg["f1"], f1_uplift=f1_up, f1_ci=[f1_lo, f1_hi],
        ks_base=mb["ks"], ks_graph=mg["ks"], ks_uplift=ks_up, ks_ci=[ks_lo, ks_hi],
        lift_base=mb["lift10"], lift_graph=mg["lift10"], lift_uplift=lift_up, lift_ci=[lift_lo, lift_hi],
        top5_features=list(zip(top5.index.tolist(), top5.to_numpy().tolist())),
        graph_in_top5=bool(graph_in_top5),
        pass_auc=bool(pass_auc), pass_f1=bool(pass_f1), pass_ks=bool(pass_ks),
        pass_lift=bool(pass_lift), accept=bool(accept),
    )

    return dict(name=name, n=cfg["n_customers"], n_edges=len(edges),
                t_gen=t_gen, t_feat=t_feat, mem_feat=mem_feat,
                t_train_b=t_train_b, t_train_g=t_train_g,
                auc_uplift=mg["auc"] - mb["auc"], accept=accept,
                graph=graph_df, model_g=model_g, metrics_record=metrics_record)


def incremental_latency_bench(graph_df, edges, n_trials=40, batch=30):
    rng = np.random.default_rng(0)
    G = build_graph(graph_df, edges)
    ids = graph_df["id"].to_numpy()
    pr = None
    latencies = []
    for _ in range(n_trials):
        new_edges = rng.choice(ids, size=(batch, 2))
        for a, b in new_edges:
            if a != b:
                G.add_edge(int(a), int(b), weight=G.get_edge_data(int(a), int(b), {"weight": 0})["weight"] + 1)
        t0 = time.perf_counter()
        pr = nx.pagerank(G, weight="weight", nstart=pr)
        latencies.append(time.perf_counter() - t0)
    return np.percentile(latencies, 95)


def run():
    print(f"[Supply Chain Finance] Model backend: {'LightGBM' if HAS_LGBM else 'sklearn HistGradientBoostingClassifier (LightGBM unavailable offline)'}")
    print(f"[Supply Chain Finance] Feature ranking: {'SHAP TreeExplainer' if HAS_SHAP else 'permutation importance (shap unavailable offline)'}")

    results = [run_scale(name, cfg) for name, cfg in SCALE_CONFIG.items()]

    print("\n=== System benchmarks ===")
    ns = np.array([r["n_edges"] for r in results], dtype=float)
    build_times = np.array([r["t_feat"] for r in results])
    mems = np.array([r["mem_feat"] for r in results])
    slope_time = np.polyfit(np.log(ns), np.log(build_times), 1)[0]
    slope_mem = np.polyfit(np.log(ns), np.log(mems), 1)[0]
    print(f"Build time vs edges (log-log slope): {slope_time:.3f} "
          f"-> {'sub-linear (OK)' if slope_time < 1 else 'NOT sub-linear'}")
    print(f"RAM peak vs edges (log-log slope):   {slope_mem:.3f} "
          f"-> {'sub-linear (OK)' if slope_mem < 1 else 'NOT sub-linear'}")

    for r in results:
        print(f"[{r['name']:6s}] n_edges={r['n_edges']:>7d}  build={r['t_feat']:.3f}s  "
              f"mem_peak={r['mem_feat']:.2f}MB  train_base={r['t_train_b']:.3f}s  "
              f"train_graph={r['t_train_g']:.3f}s")

    large = results[-1]
    _, large_edges = generate_dataset(**SCALE_CONFIG["LARGE"])
    p95_latency = incremental_latency_bench(large["graph"], large_edges)
    print(f"\nIncremental update latency p95 (LARGE, batch={30}): {p95_latency*1000:.1f} ms "
          f"-> {'PASS (<=5s)' if p95_latency <= 5 else 'FAIL'}")

    total_time_large = large["t_gen"] + large["t_feat"] + large["t_train_b"] + large["t_train_g"]
    print(f"Retraining cadence check (LARGE total pipeline time): {total_time_large:.2f}s "
          f"-> {'fits <=24h daily cycle' if total_time_large <= 86400 else 'EXCEEDS 24h'}")

    print("\nCost/uplift ratio (compute-time-per-customer / AUC-uplift), lower is better with scale:")
    for r in results:
        total_t = r["t_gen"] + r["t_feat"] + r["t_train_b"] + r["t_train_g"]
        cost_per_cust = total_t / r["n"]
        ratio = cost_per_cust / max(r["auc_uplift"], 1e-6)
        print(f"[{r['name']:6s}] cost/customer={cost_per_cust*1000:.4f}ms  "
              f"auc_uplift={r['auc_uplift']:+.4f}  cost_per_uplift={ratio*1000:.4f}ms")

    report = dict(
        problem="Supply Chain Finance — dự đoán rủi ro default của SME dựa trên vị trí trong chuỗi cung ứng",
        model_backend="LightGBM" if HAS_LGBM else "sklearn HistGradientBoostingClassifier",
        importance_method="SHAP" if HAS_SHAP else "permutation importance",
        scales=[r["metrics_record"] for r in results],
        benchmarks=dict(
            slope_time=slope_time, slope_mem=slope_mem,
            per_scale=[dict(name=r["name"], n=r["n"], n_edges=r["n_edges"], t_feat=r["t_feat"],
                            mem_feat=r["mem_feat"], t_train_b=r["t_train_b"], t_train_g=r["t_train_g"])
                       for r in results],
            p95_latency_ms=p95_latency * 1000,
            retrain_total_s=total_time_large,
            cost_uplift=[dict(name=r["name"],
                               cost_per_customer_ms=(r["t_gen"] + r["t_feat"] + r["t_train_b"] + r["t_train_g"]) / r["n"] * 1000,
                               auc_uplift=r["auc_uplift"],
                               cost_per_uplift_ms=((r["t_gen"] + r["t_feat"] + r["t_train_b"] + r["t_train_g"]) / r["n"]
                                                   / max(r["auc_uplift"], 1e-6) * 1000))
                         for r in results],
        ),
    )
    return report


if __name__ == "__main__":
    report = run()
    with open("metrics_supply_chain.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved metrics_supply_chain.json")
