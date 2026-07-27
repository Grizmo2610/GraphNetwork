import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.cluster import pair_confusion_matrix

SCALE_CONFIG = {
    "SMALL":  dict(n_persons=800),
    "MEDIUM": dict(n_persons=3200),
    "LARGE":  dict(n_persons=12800),
}


def generate(n_persons, seed=42):
    rng = np.random.default_rng(seed)
    N = n_persons
    true_id = np.arange(N)

    has_dup = rng.random(N) < 0.14
    n_records_per = 1 + has_dup.astype(int)
    total_records = int(n_records_per.sum())

    rec_true_id = np.repeat(true_id, n_records_per)
    local_j = np.concatenate([np.arange(k) for k in n_records_per]) if N > 0 else np.array([], dtype=int)
    is_second_record = local_j == 1

    base_phone = rng.integers(1_000_000, 9_999_999, N).astype(str)
    base_addr = rng.integers(10_000_000, 99_999_999, N).astype(str)
    base_nid = rng.integers(100_000_000, 999_999_999, N).astype(str)

    phone = base_phone[rec_true_id].copy()
    addr = base_addr[rec_true_id].copy()
    nid = base_nid[rec_true_id].copy()
    name_variant = np.zeros(total_records, dtype=bool)

    dup_idx = np.where(is_second_record)[0]
    n_dup = len(dup_idx)
    shares_phone = rng.random(n_dup) < 0.55
    shares_addr = rng.random(n_dup) < 0.45
    nid_typo = rng.random(n_dup) < 0.4
    name_variant[dup_idx] = rng.random(n_dup) < 0.6

    new_phone = rng.integers(1_000_000, 9_999_999, n_dup).astype(str)
    phone[dup_idx] = np.where(shares_phone, phone[dup_idx], new_phone)
    new_addr = rng.integers(10_000_000, 99_999_999, n_dup).astype(str)
    addr[dup_idx] = np.where(shares_addr, addr[dup_idx], new_addr)

    nid_arr = nid[dup_idx]
    digit_pos = rng.integers(0, 9, n_dup)
    new_digit = rng.integers(0, 10, n_dup).astype(str)
    typo_nid = np.array([s[:p] + d + s[p+1:] for s, p, d in zip(nid_arr, digit_pos, new_digit)])
    nid[dup_idx] = np.where(nid_typo, typo_nid, nid_arr)

    return pd.DataFrame({
        "record_id": np.arange(total_records), "true_id": rec_true_id,
        "phone": phone, "address": addr, "national_id": nid, "name_variant": name_variant,
    })


def baseline_match(df):
    return df.groupby("national_id").ngroup().to_numpy()


def graph_match(df):
    G = nx.Graph()
    G.add_nodes_from(df["record_id"])
    for col in ["phone", "address"]:
        for _, grp in df.groupby(col):
            ids = grp["record_id"].to_numpy()
            for i in range(1, len(ids)):
                G.add_edge(ids[0], ids[i])
    comp_id = np.zeros(len(df), dtype=int)
    for cid, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_id[node] = cid
    return comp_id


def prf_from_labels(true, pred):
    m = pair_confusion_matrix(true, pred)
    tn, fp, fn, tp = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def bootstrap_prf_ci(true, pred_base, pred_graph, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    n = len(true)
    diffs = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        pb = prf_from_labels(true[idx], pred_base[idx])
        pg = prf_from_labels(true[idx], pred_graph[idx])
        diffs[b] = [pg[i] - pb[i] for i in range(3)]
    return diffs.mean(axis=0), np.percentile(diffs, 2.5, axis=0), np.percentile(diffs, 97.5, axis=0)


def run_scale(name, cfg):
    df = generate(**cfg)
    true = df["true_id"].to_numpy()
    pred_base = baseline_match(df)
    pred_graph = graph_match(df)

    p_b, r_b, f_b = prf_from_labels(true, pred_base)
    p_g, r_g, f_g = prf_from_labels(true, pred_graph)
    uplift, lo, hi = bootstrap_prf_ci(true, pred_base, pred_graph)

    n_true_dup = int((df.groupby("true_id").size() > 1).sum())
    return dict(
        name=name, n_records=len(df), n_true_persons=int(cfg["n_persons"]), n_duplicated_persons=n_true_dup,
        precision_base=float(p_b), precision_graph=float(p_g),
        recall_base=float(r_b), recall_graph=float(r_g),
        f1_base=float(f_b), f1_graph=float(f_g),
        f1_uplift=float(uplift[2]), f1_ci=[float(lo[2]), float(hi[2])], pass_f1=bool(lo[2] > 0),
        recall_uplift=float(uplift[1]), recall_ci=[float(lo[1]), float(hi[1])], pass_recall=bool(lo[1] > 0),
        precision_uplift=float(uplift[0]), precision_ci=[float(lo[0]), float(hi[0])], pass_precision=bool(lo[0] > 0),
    )


def run():
    print("[Customer Relationship / Identity Resolution] exact-match baseline vs graph connected-components")
    scales = []
    for name, cfg in SCALE_CONFIG.items():
        r = run_scale(name, cfg)
        scales.append(r)
        print(f"{name}: F1 base={r['f1_base']:.4f} graph={r['f1_graph']:.4f} uplift={r['f1_uplift']:+.4f} "
              f"CI[{r['f1_ci'][0]:+.4f},{r['f1_ci'][1]:+.4f}] -> {'PASS' if r['pass_f1'] else 'FAIL'}  "
              f"(precision {r['precision_base']:.4f}->{r['precision_graph']:.4f}, "
              f"recall {r['recall_base']:.4f}->{r['recall_graph']:.4f})")
    return dict(problem="Customer Relationship — hợp nhất định danh (identity resolution) qua liên kết đồ thị", scales=scales)


if __name__ == "__main__":
    report = run()
    with open("metrics_identity_resolution.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved metrics_identity_resolution.json")
