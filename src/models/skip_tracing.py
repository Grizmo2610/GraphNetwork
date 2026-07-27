import json
import numpy as np
import pandas as pd

SCALE_CONFIG = {
    "SMALL":  dict(n_customers=800),
    "MEDIUM": dict(n_customers=3200),
    "LARGE":  dict(n_customers=12800),
}


def generate(n_customers, seed=42):
    rng = np.random.default_rng(seed)
    N = n_customers

    n_phones = rng.integers(1, 4, N)
    total_ph = n_phones.sum()
    ph_active = rng.random(total_ph) < 0.72
    cust_idx_ph = np.repeat(np.arange(N), n_phones)
    ph_df = pd.DataFrame({"cust": cust_idx_ph, "active": ph_active})
    own_phone_active = ph_df.groupby("cust")["active"].any().reindex(range(N), fill_value=False).to_numpy()
    lost_contact = ~own_phone_active

    address_active = rng.random(N) < 0.55

    n_related = rng.integers(0, 3, N)
    n_persons_per = 1 + n_related
    total_persons = int(n_persons_per.sum())
    cust_idx_p = np.repeat(np.arange(N), n_persons_per)
    local_j = np.concatenate([np.arange(k) for k in n_persons_per]) if N > 0 else np.array([], dtype=int)
    is_guarantor = local_j == 0
    person_phone_active = rng.random(total_persons) < 0.75
    shares_address = rng.random(total_persons) < 0.35

    p_df = pd.DataFrame({"cust": cust_idx_p, "is_guarantor": is_guarantor,
                         "phone_active": person_phone_active, "shares_address": shares_address})
    reach_via_person = p_df.groupby("cust")["phone_active"].any().reindex(range(N), fill_value=False).to_numpy()
    reach_via_guarantor = (p_df[p_df.is_guarantor].groupby("cust")["phone_active"].any()
                          .reindex(range(N), fill_value=False).to_numpy())
    reach_via_shared_address = p_df.groupby("cust")["shares_address"].any().reindex(range(N), fill_value=False).to_numpy()

    baseline_reach = address_active
    graph_reach = address_active | reach_via_person | reach_via_shared_address

    return pd.DataFrame({
        "id": np.arange(N), "lost_contact": lost_contact, "address_active": address_active,
        "reach_via_guarantor": reach_via_guarantor, "reach_via_person": reach_via_person,
        "reach_via_shared_address": reach_via_shared_address,
        "baseline_reach": baseline_reach, "graph_reach": graph_reach,
    })


def bootstrap_ci(mask_vals_base, mask_vals_graph, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mask_vals_base)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = mask_vals_graph[idx].mean() - mask_vals_base[idx].mean()
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def run_scale(name, cfg):
    df = generate(**cfg)
    lc = df[df.lost_contact].reset_index(drop=True)
    base_vals = lc["baseline_reach"].to_numpy().astype(float)
    graph_vals = lc["graph_reach"].to_numpy().astype(float)

    reach_base = base_vals.mean()
    reach_graph = graph_vals.mean()
    uplift, lo, hi = bootstrap_ci(base_vals, graph_vals)

    via_guarantor = lc["reach_via_guarantor"].mean()
    via_person = lc["reach_via_person"].mean()
    via_address = lc["reach_via_shared_address"].mean()

    return dict(
        name=name, n=cfg["n_customers"], n_lost_contact=int(len(lc)),
        lost_contact_rate=float(df["lost_contact"].mean()),
        reach_base=float(reach_base), reach_graph=float(reach_graph),
        uplift=float(uplift), ci=[float(lo), float(hi)], pass_uplift=bool(lo > 0),
        residual_manual_review=float(1 - reach_graph),
        channel_breakdown=dict(guarantor_phone=float(via_guarantor), any_person_phone=float(via_person),
                               shared_address=float(via_address)),
    )


def run():
    print("[Skip Tracing] reach-rate baseline (direct contact only) vs graph (indirect channels)")
    scales = []
    for name, cfg in SCALE_CONFIG.items():
        r = run_scale(name, cfg)
        scales.append(r)
        print(f"{name}: lost_contact={r['n_lost_contact']}/{r['n']} ({r['lost_contact_rate']:.1%})  "
              f"reach base={r['reach_base']:.3f} graph={r['reach_graph']:.3f} uplift={r['uplift']:+.3f} "
              f"CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] -> {'PASS' if r['pass_uplift'] else 'FAIL'}  "
              f"residual manual review={r['residual_manual_review']:.1%}")
    return dict(problem="Skip Tracing — tìm kênh liên hệ gián tiếp cho khách hàng mất liên lạc", scales=scales)


if __name__ == "__main__":
    report = run()
    with open("metrics_skip_tracing.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved metrics_skip_tracing.json")
