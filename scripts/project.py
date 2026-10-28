"""Reduce real LLM hidden states to 3D and export the web asset.

Two layouts per layer, both computed from the same real activations:

  pca    - one COMMON linear basis for every layer: PCA over the union of all
           layers' per-layer-centred, unit-RMS states. Normalising each layer to
           unit RMS first keeps the frame balanced across depth (raw norms grow
           ~1000x, so an unnormalised union is dominated by the last layers). One
           shared frame means the layer slider animates the representation in a
           fixed coordinate system instead of an arbitrary rotation. Axis shares
           are real and labelled.

  layout - UMAP (cosine, k=15) per layer, so semantic neighbourhoods land near
           each other. Distances in this layout are not metric and each layer is
           laid out independently - positions are not comparable across layers.
           It exists because the linear projection measurably hides most of the
           cluster structure; both layouts carry their own measured separation so
           the difference is quantified, not asserted.

Exports
  coords.pca[l], coords.layout[l]  3D coordinates per layer (unit RMS)
  rms[l], spread[l]                real per-layer norms (drives "true scale")
  varRatio[l]                      share of a layer's variance inside the 3 PCA axes
  purityFull[l]                    semantics: 5-NN sharing the topic, in the full 896-d space
  purityPca[l], purityLayout[l]    the same measure taken inside each 3D layout
  traj                             real greedy generation path, projected into the PCA basis
"""
import json, os, datetime, numpy as np
import umap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 5


def purity(P, tindex, k=K):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    S = Pn @ Pn.T
    np.fill_diagonal(S, -9)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return float((tindex[idx] == tindex[:, None]).mean())


def unit_rms(X):
    X = X - X.mean(axis=0, keepdims=True)
    return X / (np.sqrt((X ** 2).sum(1).mean()) + 1e-9)


FIT = 0.8   # half-extent every layer is fitted into, so framing holds at any depth


def fit_layer(P):
    """Centre one layer's geometry and scale it into a cube of half-extent FIT.
    Returns (fitted, centre, scale) so other geometry can take the same transform.
    Framing is the reason: raw norms grow ~1000x with depth, and one shared scale
    would leave the early layers a dot in the middle of the frame."""
    c = (P.max(0) + P.min(0)) / 2.0
    h = float(np.abs(P - c).max())
    if h == 0:
        return P - c, c, 1.0
    return (P - c) / h * FIT, c, h / FIT


def fit_layers(X):
    return np.stack([fit_layer(X[l])[0] for l in range(X.shape[0])])


def main():
    d = np.load(os.path.join(ROOT, "data", "hidden.npz"))
    H = d["hidden"].astype(np.float64)                       # [L, N, D]
    L, N, D = H.shape
    meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
    tset = sorted(set(meta["topics"]))
    tindex = np.array([tset.index(t) for t in meta["topics"]])

    Hc = H - H.mean(axis=1, keepdims=True)
    rms = np.sqrt((Hc ** 2).sum(-1).mean(axis=1))
    Hn = Hc / rms[:, None, None]

    S = np.concatenate([Hn[l] for l in range(L)], axis=0)
    _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    B = Vt[:3]
    print("PCA basis singular values:", np.round(sv[:6], 1),
          " top3 share of union:", np.round((sv[:3] ** 2) / (sv ** 2).sum(), 3))

    pca_raw = np.stack([Hn[l] @ B.T for l in range(L)])
    pca = fit_layers(pca_raw)
    lay = np.zeros_like(pca)
    varRatio = np.zeros(L); pFull = np.zeros(L); pPca = np.zeros(L); pLay = np.zeros(L)
    for l in range(L):
        Xn = Hn[l]
        varRatio[l] = ((Xn @ B.T) ** 2).sum() / (Xn ** 2).sum()
        U = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.12, metric="cosine",
                      random_state=3, init="spectral", n_epochs=500).fit_transform(Xn)
        lay[l] = U
        pFull[l] = purity(Xn, tindex)
        pPca[l] = purity(pca[l], tindex)
        print(f"  layer {l:2d}  pca-frame={varRatio[l]*100:5.1f}%  spread={rms[l]/rms.max():5.3f}"
              f"  separation full={pFull[l]*100:4.0f}% pca={pPca[l]*100:4.0f}%", flush=True)

    lay = fit_layers(lay)
    for l in range(L):     # measured on the exact geometry the viewer shows
        pLay[l] = purity(lay[l], tindex)
    print("layout separation:", [f"{v*100:.0f}%" for v in pLay])

    steps = np.load(os.path.join(ROOT, "data", "traj.npz"))
    # the trajectory lives on the final layer's PCA geometry, so it takes the
    # same centre/scale transform, then stays put while the cloud morphs
    Tp = ((steps["steps"].astype(np.float64) - H[-1].mean(axis=0, keepdims=True)) / rms[-1]) @ B.T
    _, pca_c, pca_h = fit_layer(pca_raw[-1])
    Tp = (Tp - pca_c) / pca_h * FIT
    print("pca extent:", np.round(pca.reshape(-1, 3).min(0), 2), np.round(pca.reshape(-1, 3).max(0), 2))
    print("layout extent:", np.round(lay.reshape(-1, 3).min(0), 2), np.round(lay.reshape(-1, 3).max(0), 2))

    out = {
        "model": str(d["model"][0]),
        "nSentences": N, "nLayers": L, "hiddenSize": int(D),
        "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "topics": tset,
        "sentences": [{"t": t, "y": y} for t, y in zip(meta["texts"], meta["topics"])],
        "coords": {"pca": np.round(pca, 4).tolist(), "layout": np.round(lay, 4).tolist()},
        "pcVar": np.round((sv[:3] ** 2) / (sv ** 2).sum(), 5).tolist(),
        "rms": np.round(rms, 4).tolist(),
        "spread": np.round(rms / rms.max(), 5).tolist(),
        "varRatio": np.round(varRatio, 5).tolist(),
        "purityFull": np.round(pFull, 4).tolist(),
        "purityPca": np.round(pPca, 4).tolist(),
        "purityLayout": np.round(pLay, 4).tolist(),
        "traj": {"prompt": str(steps["prompt"][0]), "text": str(steps["text"][0]),
                 "coords": np.round(Tp, 4).tolist()},
    }
    dest = os.path.join(ROOT, "web", "data", "latent.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"))
    print(f"-> {dest} ({os.path.getsize(dest)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
