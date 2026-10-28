"""Compare candidate 3D bases numerically before picking one for the viewer.

For each candidate: build the common basis over all layers, project, and measure
  var3       - share of that layer's variance captured by the plotted 3 axes
  purity3d   - semantic purity measured IN the 3D projection (what you actually see)
so the choice is made on evidence, not on which rendering looks nicer.
"""
import json, os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "data", "hidden.npz"))
H = d["hidden"].astype(np.float64)
L, N, D = H.shape
meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
tset = sorted(set(meta["topics"]))
tindex = np.array([tset.index(t) for t in meta["topics"]])
k = 5

def purity_of(P):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    S = Pn @ Pn.T
    np.fill_diagonal(S, -9)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return (tindex[idx] == tindex[:, None]).mean()

def run(name, prep):
    pre = [prep(H[l]) for l in range(L)]                     # per-layer prepared + centred
    S = np.concatenate(pre, axis=0)
    _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    B = Vt[:3]
    v3 = np.array([((pre[l] @ B.T) ** 2).sum() / (pre[l] ** 2).sum() for l in range(L)])
    p3 = np.array([purity_of(pre[l] @ B.T) for l in range(L)])
    print(f"\n== {name} == pcVar(top3 of union)={np.round((sv[:3]**2)/(sv**2).sum(),3)}")
    print("   layer : " + " ".join(f"{l:4d}" for l in range(L)))
    print("   3d-var: " + " ".join(f"{v*100:4.0f}" for v in v3))
    print("   3d-pur: " + " ".join(f"{p*100:4.0f}" for p in p3))
    return p3

def centre(X):
    return X - X.mean(0, keepdims=True)

run("A raw centred", centre)
run("B unit-RMS centred", lambda X: centre(X / (np.sqrt((X - X.mean(0))**2).sum(-1).mean())))
run("C cosine then centred", lambda X: centre(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)))
run("D cosine, unit-RMS, centred", lambda X: centre(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)))
