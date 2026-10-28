# Latent Space — a 3D view of a real LLM's hidden states

A single-page WebGL explorer of the internal representation space of
`Qwen/Qwen2.5-0.5B-Instruct`. Every point is a real sentence run through the real
model; nothing is simulated, and every number in the UI is measured.

Live: https://jacoblbagent.github.io/latent-space-viz/ — local: :5205, in the
Visualizations hub at https://jacoblbagent.github.io/visualizations/.

![view](preview.png)

## What is actually plotted

208 hand-written sentences across 13 domains (animals, cooking, programming,
mathematics, weather, music, medicine, law, space, sports, finance, literature,
technology). Each sentence is pushed through the 0.5B-parameter model and the
**mean-pooled hidden state of every layer** is captured: 25 tensors of 896
dimensions each (embedding layer + 24 transformer layers).

Two layouts of the same activations are offered:

| Layout | What it is | Distances mean something? |
|---|---|---|
| `linear · PCA` | One common linear basis for all layers: PCA over the union of every layer's per-layer-centred, unit-RMS states. A single shared frame, so the layer slider animates the representation in a fixed coordinate system rather than a per-layer rotation. Axis labels show each principal axis's real share of total variance. | Yes |
| `neighbourhood · UMAP` | UMAP (cosine, 15 neighbours) per layer, so semantic neighbours land near each other. | No — neighbour structure only |

## Why two layouts

The 3D PCA picture is measurably incomplete, and the app says so instead of
hiding it. Cluster separation (the share of each sentence's 5 nearest neighbours
that share its domain) is measured in the full 896-d space and again inside each
3D layout. At layer 23 the full space separates 91% and the UMAP layout 90%, but
the linear projection only shows 34%. That gap is the point: the semantic
structure of these activations is not three-dimensional, and a linear projection
of it is a cigar, not a cluster map.

The metrics panel reports, per layer:

- **Cluster separation** — 5-NN topic purity inside the view you are looking at, and in the full space for comparison.
- **Representation spread** — mean squared norm of the centred state; it grows about 1000x from layer 0 to layer 24.
- **Variance on the three axes** (PCA view) — how much of that layer's variance the plotted axes capture.

The bar strip under those numbers is the full-space separation for all 25 layers;
click any bar to jump to that depth.

## Things worth looking at

- Layer 0-2: near-isotropic and tiny — the embedding table's directions are not yet organised by meaning.
- Layers 3-21: variance collapses into a few "rogue" directions (the three PCA axes hold ~94% of it), and topical purity plateaus around 43%.
- Layers 22-23: purity jumps to ~90% and, in the neighbourhood layout, 13 clean domain islands appear.
- Layer 24: the norm explodes to ~1000x every other layer, the signature of the huge activations feeding the output head.
- **Generation path** (toggle): the model is asked to write a sentence about the ocean; its greedy 40-token output is embedded step by step and drawn as a path through the final layer's PCA frame. The trajectory stays in one neighbourhood instead of wandering.

## Running it

```bash
./start-latent-space-viz.sh      # prints the URL, serves the repo root
```

Static files only — `index.html`, `three.min.js`, `fonts.css`, `fonts/`, `data/latent.json` —
so any static server works and GitHub Pages serves it with no build step. It must
be served over HTTP (the page fetches `data/latent.json`), not opened as `file://`.

## Rebuilding the data

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers safetensors umap-learn
.venv/bin/python scripts/extract.py    # model forward passes -> data/hidden.npz, data/traj.npz
.venv/bin/python scripts/project.py    # PCA + UMAP + metrics  -> data/latent.json
```

`scripts/basis_experiment.py` and `scripts/layout_compare.py` are the evidence
runs behind the layout choices: they compare candidate bases and layout methods
using the measured 3D separation rather than eyeballing a render.

## Files

```
data/corpus.json          the 208 sentences, grouped by domain
data/hidden.npz           [25 layers, 208 sentences, 896 dims] real hidden states
data/traj.npz             per-token final-layer states for the generation path
scripts/extract.py        model -> hidden states
scripts/project.py        hidden states -> 3D + metrics -> data/latent.json
scripts/fetch_fonts.py    self-hosts the two typefaces
index.html                the viewer (three.js r160, no build step)
three.min.js              local three.js
fonts.css, fonts/         Instrument Sans + IBM Plex Mono, latin subset
assets/favicon.svg        the hub's favicon
```

## Caveats

- 208 sentences is a small sample; the layer profiles are indicative, not a benchmark.
- UMAP layouts are stochastic per layer and computed independently, so positions
  are not comparable between layers — only within one layer.
- The `true scale` toggle uses real per-layer norms; the `normalised` toggle fits
  each layout into a fixed cube so every depth is readable. Shapes differ between
  the two modes because the former is dominated by norm growth.
- Sentence-level mean pooling is one summary of a layer; token-level geometry is not shown.
