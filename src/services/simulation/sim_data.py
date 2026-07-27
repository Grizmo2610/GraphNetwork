import numpy as np
import pandas as pd

SCALE_CONFIG = {
    "SMALL":  dict(n_customers=800,   avg_degree=8),
    "MEDIUM": dict(n_customers=3200,  avg_degree=9),
    "LARGE":  dict(n_customers=12800, avg_degree=10),
}


def generate_dataset(n_customers, avg_degree, seed=42):
    rng = np.random.default_rng(seed)
    N = n_customers
    ids = np.arange(N)

    n_comm = max(10, N // 150)
    community = rng.integers(0, n_comm, N)
    ring_comms = rng.choice(n_comm, size=max(1, int(0.12 * n_comm)), replace=False)
    comm_risk_all = np.where(np.isin(np.arange(n_comm), ring_comms),
                              rng.normal(1.7, 0.3, n_comm),
                              rng.normal(0.0, 0.3, n_comm))
    risk_per_customer = comm_risk_all[community]
    is_ring_customer = np.isin(community, ring_comms)

    debt_amount = rng.lognormal(mean=9.5, sigma=0.8, size=N) * (1 + 0.15 * is_ring_customer)
    overdue_days = np.clip(rng.normal(60, 45, N) + 25 * is_ring_customer, 0, 720)
    income = rng.lognormal(mean=8.8, sigma=0.5, size=N)
    age = rng.integers(21, 65, N)
    employment_years = rng.integers(0, 30, N)
    application_amount = debt_amount * rng.uniform(0.5, 1.5, N)

    n_phones = rng.integers(1, 4, N)
    phone_active_ratio = rng.beta(2, 1, N) * (rng.random(N) > 0.1)
    address_active = (rng.random(N) > 0.25).astype(float)

    n_alt_contacts = rng.integers(0, 4, N)
    alt_active_p = np.clip(0.6 - 0.1 * is_ring_customer, 0.1, 1.0)
    n_alt_contacts_active = rng.binomial(n_alt_contacts, alt_active_p)

    still_employed = (rng.random(N) > 0.35).astype(float)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    logit = (0.55 * z(debt_amount) + 0.75 * z(overdue_days) - 0.35 * z(income)
             - 0.25 * z(employment_years) - 0.4 * z(phone_active_ratio)
             + 1.4 * risk_per_customer + rng.normal(0, 0.6, N) - 1.1)
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    customers = pd.DataFrame({
        "id": ids, "debt_amount": debt_amount, "overdue_days": overdue_days,
        "income": income, "age": age, "employment_years": employment_years,
        "application_amount": application_amount, "n_phones": n_phones,
        "phone_active_ratio": phone_active_ratio, "address_active": address_active,
        "n_alt_contacts": n_alt_contacts, "n_alt_contacts_active": n_alt_contacts_active,
        "still_employed": still_employed, "default": default, "_true_segment": community,
    })

    n_total_edges = N * avg_degree // 2
    within_mask = rng.random(n_total_edges) < 0.85
    src = rng.integers(0, N, n_total_edges)
    tgt = np.empty(n_total_edges, dtype=int)

    comm_members = {c: ids[community == c] for c in range(n_comm)}
    src_comm = community[src]
    for c in range(n_comm):
        sel = within_mask & (src_comm == c)
        cnt = sel.sum()
        if cnt == 0:
            continue
        members = comm_members[c]
        tgt[sel] = members[rng.integers(0, len(members), cnt)]
    across_sel = ~within_mask
    tgt[across_sel] = rng.integers(0, N, across_sel.sum())

    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    edges = pd.DataFrame({"src": np.minimum(src, tgt), "tgt": np.maximum(src, tgt)})
    edges["amount"] = rng.uniform(1e5, 5e7, len(edges))
    edges = edges.groupby(["src", "tgt"], as_index=False).agg(weight=("amount", "count"),
                                                                total_amount=("amount", "sum"))

    return customers, edges
