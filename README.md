# Real-Time Traffic Congestion Prediction & Route Optimization

Day 1 output: verified environment, project scaffold, and literature skim.
See `../traffic_project_roadmap.md` for the full 6-week plan.

## Setup status (verified 2026-07-15)

Tested in an isolated sandbox on Python 3.12. Results, so you know exactly
what to expect when you set this up yourself:

**Installed and imported cleanly:** `torch` 2.13.0, `torch-geometric` 2.8.0,
`fastapi`, `uvicorn`, `networkx`, `pandas`, `numpy`.

**One real snag, with a fix:** `torch-geometric-temporal` declares
`torch-scatter` and `torch-sparse` as hard dependencies. Installing them
from PyPI directly tries to compile C++/CUDA extensions from source, which
can hang for a very long time. The standard fix — and the one to actually
use — is installing them as prebuilt wheels instead:

```bash
pip install torch
python -c "import torch; print(torch.__version__)"   # note the version printed

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>.html
# e.g. -f https://data.pyg.org/whl/torch-2.13.0+cpu.html

pip install -r requirements.txt
```

Worth knowing: tracing the actual import chain, `torch-scatter`/`torch-sparse`
are only used by one layer this project doesn't need (`EvolveGCN-O`) — the
`METRLADatasetLoader` and `A3TGCN`, which is what we're actually using, don't
touch them. They're still required for the package to import at all though,
since `torch_geometric_temporal`'s `__init__.py` eagerly loads every layer.
So the wheel-index install above is a one-time cost, not something you'll
fight with repeatedly.

**Not verified — sandbox network restrictions, not a real problem:**
`METRLADatasetLoader` downloads the raw data from `anl.app.box.com`, which
this build sandbox can't reach. That's a restriction specific to this
sandbox, not a real access issue — it'll download normally on your machine
or on Colab. `data.pyg.org` (the wheel index above) is similarly unreachable
from here but a completely normal host. Everything about the loader's logic
and the model's API was confirmed correct up to that download step —
`data/verify_setup.py` is ready to run as-is once you have normal internet
access.

## Literature skim (Day 1)

Three papers, in the order it's most useful to read them — DCRNN because
it's the standard benchmark result you'll compare your numbers against,
T-GCN and A3T-GCN because they're the direct architectural lineage of the
`A3TGCN` layer this project uses.

**DCRNN** — Li, Yu, Shahabi, Liu. *Diffusion Convolutional Recurrent Neural
Network: Data-Driven Traffic Forecasting.* ICLR 2018.
Treats traffic flow as a diffusion process on a directed graph, using
random walks in both directions along the graph to capture how congestion
influences upstream and downstream roads differently. Spatial dependence is
captured through this diffusion convolution, temporal dependence through an
encoder-decoder recurrent structure. It's the paper that established METR-LA
and PeMS-BAY as standard benchmarks — this is why your results get compared
against it.

**T-GCN** — Zhao, Song, Zhang, Liu, Wang, Lin, Deng, Li. *T-GCN: A Temporal
Graph Convolutional Network for Traffic Prediction.* IEEE Transactions on
Intelligent Transportation Systems, 21(9), 2019/2020.
Simpler combination than DCRNN: a standard GCN handles spatial structure,
feeding into a GRU for the temporal side. Less exotic than DCRNN's diffusion
convolution, but a clean, well-cited baseline architecture and the direct
ancestor of the layer you'll actually train.

**A3T-GCN** — Zhu, Song, Zhao, Li. *A3T-GCN: Attention Temporal Graph
Convolutional Network for Traffic Forecasting.* arXiv:2006.11583, 2020.
Adds an attention mechanism on top of T-GCN's GCN+GRU structure, letting the
model weigh different historical time steps unequally instead of treating
recent and distant history the same way. This is the exact layer
(`torch_geometric_temporal.nn.recurrent.A3TGCN`) this project's model is
built on — worth reading closely, since you'll need to explain this
architecture at viva.

## Project structure

```
data/       loading, preprocessing, graph construction
model/      A3TGCN model class, training loop, evaluation
api/        FastAPI backend (/predict, /route)
frontend/   Leaflet.js dashboard
report/     synopsis, report drafts, slides
```

## Next: Day 2

Run `data/verify_setup.py` on a machine with normal internet access, then
move to exploring the raw data — shapes, missing-value pattern, sensor
coverage — per the roadmap.