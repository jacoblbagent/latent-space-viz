"""Retune UMAP spacing on the current corpus.

Two things must hold at once at the layers where clusters are real (21-24):
  purity      - the layout still shows the structure that exists (5-NN share the domain)
  inflation   - the empty space between domains is not invented

inflation = (gap ratio in the 3D layout) / (gap ratio in the full space), and
1.00x means faithful. Reported per layer, not averaged, so a config that looks
fine on average but blows up on the money layers cannot win.
"""
import json, os, sys, numpy as np, umap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "data", "hidden.npz"))
H = d["hidden"].astype(np.float64)
meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
topics = meta["topics"]; N = H.shape[1]
Hc = H - H.mean(axis=1, keepdims=True)
Hn = Hc / np.sqrt((Hc ** 2).sum(-1).mean(axis=1))[:, None, None]
tindex = np.array([sorted(set(topics)).index(t) for t in topics])
same = tindex[:, None] == tindex[None, :]
LAYERS = [20, 21, 22, 23, 24]


def stats(P, metric):
    if metric == "cosine":
        X = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); Dm = 1.0 - X @ X.T
    else:
        Dm = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    Dm = Dm.copy(); np.fill_diagonal(Dm, np.inf)
    gap = np.mean([Dm[i][~same[i]].min() for i in range(N)]) / np.mean(
        [np.mean(Dm[i][same[i] & (np.arange(N) != i)]) for i in range(N)])
    idx = np.argpartition(Dm, 5, axis=1)[:, :5]
    return float(gap), float(np.mean([np.mean([topics[j] == topics[i] for j in idx[i]]) for i in range(N)]))


full = {l: stats(Hn[l], "cosine")[0] for l in LAYERS}
print(f"{N} points, {len(set(topics))} domains")
print("full-space gap ratio:", {l: round(full[l], 2) for l in LAYERS})
print(f"\n{'nn':>4} {'md':>5} | " + " | ".join(f"L{l}: pur / infl" for l in LAYERS) + "  |  min-pur  max-infl")
rows = []
for nn in (30, 60, 100, 150):
    for md in (0.4, 0.6, 0.8):
        cells, purs, infls = [], [], []
        for l in LAYERS:
            U = np.asarray(umap.UMAP(n_components=3, n_neighbors=nn, min_dist=md, metric="cosine",
                                     random_state=3, init="spectral", n_epochs=500).fit_transform(Hn[l]))
            g, p = stats(U, "euclid")
            infl = g / full[l]
            purs.append(p); infls.append(infl)
            cells.append(f"{p*100:3.0f}% / {infl:4.2f}x")
        rows.append((nn, md, min(purs), max(infls)))
        print(f"{nn:4d} {md:5.2f} | " + " | ".join(cells) + f"  |  {min(purs)*100:5.1f}%  {max(infls):5.2f}x")

print("\nconfigs with max inflation <= 1.35x, best min-purity first:")
for r in sorted([x for x in rows if x[3] <= 1.35], key=lambda x: -x[2])[:4]:
    print(f"   nn={r[0]:4d} min_dist={r[1]:.2f}  min-purity={r[2]*100:.1f}%  max-inflation={r[3]:.2f}x")
