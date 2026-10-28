"""Pick the layout method on evidence: my hand-rolled neighbour layout vs UMAP, at
one ambiguous middle layer and at the layer where clusters are real. Metric = the
same 5-NN topic purity used everywhere else, measured inside the 3D result.
"""
import json, os, time, numpy as np
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("proj", os.path.join(ROOT, "scripts", "project.py"))
proj = importlib.util.module_from_spec(spec); spec.loader.exec_module(proj)

d = np.load(os.path.join(ROOT, "data", "hidden.npz"))
H = d["hidden"].astype(np.float64)
meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
tset = sorted(set(meta["topics"]))
tindex = np.array([tset.index(t) for t in meta["topics"]])
Hc = H - H.mean(axis=1, keepdims=True)
rms = np.sqrt((Hc ** 2).sum(-1).mean(axis=1))
Hn = Hc / rms[:, None, None]
S = np.concatenate([Hn[l] for l in range(H.shape[0])], axis=0)
B = np.linalg.svd(S, full_matrices=False)[2][:3]

import umap
for l in [10, 16, 20, 22, 23, 24]:
    X = Hn[l]
    pca3 = X @ B.T
    nbrs, _ = proj.knn_index(X, 8)
    t0 = time.time()
    mine = proj.layout(pca3, nbrs)
    t1 = time.time()
    um = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.12, metric="cosine",
                   random_state=3, init="spectral", n_epochs=500).fit_transform(X)
    t2 = time.time()
    print(f"layer {l:2d}  full={proj.purity(X, tindex)*100:4.0f}%"
          f"  pca3d={proj.purity(pca3, tindex)*100:4.0f}%"
          f"  mine={proj.purity(mine, tindex)*100:4.0f}% ({t1-t0:.1f}s)"
          f"  umap={proj.purity(um, tindex)*100:4.0f}% ({t2-t1:.1f}s)")
