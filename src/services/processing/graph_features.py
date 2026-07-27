import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp


def build_graph(customers, edges):
    G = nx.Graph()
    G.add_nodes_from(customers["id"].to_numpy())
    G.add_weighted_edges_from(edges[["src", "tgt", "weight"]].itertuples(index=False, name=None))
    return G


def _sparse_adj(customers, edges):
    n = len(customers)
    rows = np.concatenate([edges["src"].to_numpy(), edges["tgt"].to_numpy()])
    cols = np.concatenate([edges["tgt"].to_numpy(), edges["src"].to_numpy()])
    w = np.concatenate([edges["weight"].to_numpy(), edges["weight"].to_numpy()])
    return sp.csr_matrix((w, (rows, cols)), shape=(n, n))


def compute_features(G, customers, edges):
    n = len(customers)
    nodes = customers["id"].to_numpy()

    degree = np.array([G.degree(i) for i in nodes], dtype=float)
    weighted_degree = np.array([G.degree(i, weight="weight") for i in nodes], dtype=float)
    pagerank = nx.pagerank(G, weight="weight")
    pagerank = np.array([pagerank.get(i, 0.0) for i in nodes])
    k_sample = int(np.clip(4000 / np.sqrt(n), 15, 100))
    betweenness = nx.betweenness_centrality(G, k=k_sample, weight=None, seed=42)
    betweenness = np.array([betweenness.get(i, 0.0) for i in nodes])
    clustering = nx.clustering(G)
    clustering = np.array([clustering.get(i, 0.0) for i in nodes])

    comms = list(nx.algorithms.community.label_propagation_communities(G))
    comm_id = np.zeros(n, dtype=int)
    for cid, members in enumerate(comms):
        for m in members:
            comm_id[m] = cid

    hub_thresh = np.quantile(degree, 0.98)
    hubs = nodes[degree >= hub_thresh]
    H = G.copy()
    H.add_node("__HUB__")
    for h in hubs:
        H.add_edge("__HUB__", h)
    dist_from_hub = nx.single_source_shortest_path_length(H, "__HUB__")
    distance_to_hub = np.array([max(dist_from_hub.get(i, 99) - 1, 0) for i in nodes], dtype=float)

    adj = _sparse_adj(customers, edges)
    deg_safe = np.maximum(degree, 1)
    neighbor_mean_debt = (adj @ customers["debt_amount"].to_numpy()) / deg_safe
    neighbor_mean_overdue = (adj @ customers["overdue_days"].to_numpy()) / deg_safe
    row_max = np.asarray(adj.max(axis=1).todense()).flatten()
    row_sum = np.asarray(adj.sum(axis=1)).flatten()
    concentration = row_max / np.maximum(row_sum, 1e-9)

    feats = pd.DataFrame({
        "id": nodes, "degree": degree, "weighted_degree": weighted_degree,
        "pagerank": pagerank, "betweenness": betweenness, "clustering": clustering,
        "detected_community": pd.Categorical(comm_id), "distance_to_hub": distance_to_hub,
        "neighbor_mean_debt": neighbor_mean_debt, "neighbor_mean_overdue": neighbor_mean_overdue,
        "concentration": concentration,
    })
    return feats


GRAPH_FEATURE_COLS = ["degree", "weighted_degree", "pagerank", "betweenness", "clustering",
                      "detected_community", "distance_to_hub", "neighbor_mean_debt",
                      "neighbor_mean_overdue", "concentration"]
