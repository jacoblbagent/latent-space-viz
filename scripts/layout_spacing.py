"""Why do the clusters look so far apart, and so sparse?

Two separate questions, measured separately:

  * sparsity  - how many points each domain actually has (a corpus fact).
  * the moat  - is the empty space between islands real, or a UMAP artefact?
                Measured as a GAP RATIO: mean distance from a point to the
                nearest point of ANOTHER domain, divided by the mean radius of
                its own domain. Computed in the full 896-d space and again inside
                candidate 3D layouts, so the layout's exaggeration is a number
                rather than an impression.
"""
import json, os, numpy as np
import umap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "data", "hidden.npz"))
H = d["hidden"].astype(np.float64)
meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
topics = meta["topics"]
tset = sorted(set(topics))
L, N, D = H.shape
print(f"{N} sentences, {len(tset)} domains -> {N // len(tset)} points per domain")


def gap_ratio(P, metric="cosine"):
    """mean(nearest other-domain distance) / mean(within-domain radius)"""
    if metric == "cosine":
        X = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        Dm = 1.0 - X @ X.T
    else:
        Dm = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    same = np.array([[a == b for b in topics] for a in topics])
    np.fill_diagonal(Dm, np.inf)
    near_other = np.array([Dm[i][~same[i]].min() for i in range(N)])
    within = np.array([np.mean(Dm[i][same[i] & (np.arange(N) != i)]) for i in range(N)])
    return float(near_other.mean() / within.mean()), float(within.mean()), float(near_other.mean())


Hc = H - H.mean(axis=1, keepdims=True)
rms = np.sqrt((Hc ** 2).sum(-1).mean(axis=1))
Hn = Hc / rms[:, None, None]

for layer in (12, 22, 23):
    X = Hn[layer]
    gr, w, o = gap_ratio(X, "cosine")
    print(f"\nlayer {layer}  FULL 896-d space: within={w:.4f} nearest-other={o:.4f} GAP RATIO={gr:.2f}x")
    for nn in (15, 30, 60):
        for md in (0.12, 0.35, 0.6):
            U = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=md, metric="cosine",
                          random_state=3, init="spectral", n_epochs=500).fit_transform(X)
            g3, w3, o3 = gap_ratio(np.asarray(U), "euclid")
            # local neighbourhood quality: share of 5 nearest neighbours sharing the domain
            Dm = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
            np.fill_diagonal(Dm, np.inf)
            idx = np.argpartition(Dm, 5, axis=1)[:, :5]
            pur = float(np.mean([np.mean([topics[j] == topics[i] for j in idx[i]]) for i in range(N)]))
            print(f"   UMAP nn={nn:2d} min_dist={md:.2f}  GAP RATIO={g3:5.2f}x  (exaggeration {g3/gr:4.1f}x)  purity={pur*100:4.0f}%")
