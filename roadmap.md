# Real-Time Traffic Congestion Prediction — Day-by-Day Roadmap

**Calibrated for:** ~1.5 months (6 weeks / 42 days), solo work, ~3–5 hrs/day, free-tier GPU (Colab is fine)

**One change from the earlier plan:** `torch_geometric_temporal` ships a built-in METR-LA loader (`METRLADatasetLoader`) that hands you the 207-sensor graph — nodes, edges built from real sensor distances, and windowed input/output tensors — in about two lines of code. That replaces the OSMnx-to-sensor matching step from the original plan and removes several days of fiddly work. Routing runs over this same 207-sensor graph (highway-segment level) rather than a full street-level OSM road network. This is a legitimate scope choice, not a shortcut to hide — say so explicitly in your report's scope/limitations section.

**How to use this:** check off each day's boxes as you go. Every week ends with a buffer day — use it for slack, not new work, unless you finished early. Week 3 (training) is the most likely to overflow, so its buffer is the one to protect most.

---

## Week 1 — Setup, Data, Synopsis

**Day 1**
- [ ] Set up Python env (3.10+), install `torch`, `torch-geometric`, `torch-geometric-temporal`, `fastapi`, `networkx`, `pandas`
- [ ] Create GitHub repo + folder structure (`data/`, `model/`, `api/`, `frontend/`, `report/`)
- [ ] Skim 2–3 papers (DCRNN, T-GCN, A3TGCN) — just abstracts, architecture diagrams, and results tables

**Day 2**
- [ ] `from torch_geometric_temporal.dataset import METRLADatasetLoader` → load METR-LA
- [ ] Explore raw shapes: node count (207), feature count, time range, missing-value pattern
- [ ] Skim the official `a3tgcn_for_traffic_forecasting.ipynb` notebook in the `pytorch_geometric_temporal` GitHub repo — it's the closest existing reference to this exact project

**Day 3**
- [ ] Write project synopsis/abstract using now-confirmed dataset stats (207 sensors, LA highways, 5-min intervals, 4 months of data)
- [ ] Submit synopsis for department approval if required

**Day 4**
- [ ] Build windowed dataset: `loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)` (12 steps = 1 hr history → 1 hr forecast)
- [ ] Inspect a few samples — confirm `x`, `edge_index`, `y` shapes make sense

**Day 5**
- [ ] Chronological train/val/test split (70/10/20 — standard for METR-LA)
- [ ] Wrap in batches ready for training

**Day 6**
- [ ] Normalize speed values (z-score); store mean/std for de-normalizing predictions later
- [ ] Pull sensor lat/lon coordinates (`graph_sensor_locations.csv` from the original DCRNN repo) — you'll need these in Week 5 for the map

**Day 7 — Buffer / catch-up**
- [ ] Confirm synopsis approval status; resolve anything blocking Week 2

**✅ End of Week 1:** data pipeline running end-to-end, synopsis submitted, sensor coordinates saved.

---

## Week 2 — Model Architecture

**Day 8**
- [ ] Study the `A3TGCN` (or `TGCN`) recurrent layer from `torch_geometric_temporal` — read through the reference notebook line by line

**Day 9**
- [ ] Decide: hand-rolled GCN+LSTM vs the library's recurrent layer. Given the timeline, build your own training loop and forecasting head *around* `A3TGCN` — still substantial original engineering, much less risk

**Day 10**
- [ ] Code the full model class (A3TGCN core → linear output head)
- [ ] Verify forward pass shape on one dummy batch

**Day 11**
- [ ] Set up loss (MAE), optimizer (Adam), training loop with per-epoch logging and checkpoint saving

**Day 12**
- [ ] Dry run: 2–3 epochs on a small data subset — confirm loss decreases, no NaNs, no shape errors

**Day 13**
- [ ] Build the evaluation function: MAE / RMSE / MAPE on de-normalized values, reported separately for 15/30/60-min horizons (so you can benchmark against published numbers later)

**Day 14 — Buffer / catch-up**

**✅ End of Week 2:** model architecture coded and verified with a dry run, ready for full training.

---

## Week 3 — Training & Tuning (highest-risk week)

**Day 15**
- [ ] Launch full training run (Colab GPU if no local GPU) — this can run for hours, start it early in the day
- [ ] Log train/val loss per epoch

**Day 16**
- [ ] Check convergence. If loss plateaus early or explodes: adjust learning rate, add gradient clipping, re-check normalization

**Day 17**
- [ ] Continue or re-run training with fixes
- [ ] Sanity-check current MAE against published METR-LA benchmarks (ballpark comparison, not an exact-match expectation)

**Day 18**
- [ ] **If stalled:** cut epochs or reduce model size / sensor count to guarantee a working result over a "perfect" one
- [ ] **If on track:** light hyperparameter tuning (hidden size, number of layers, learning rate)

**Day 19**
- [ ] Lock in the best checkpoint
- [ ] Run full test-set evaluation; record final MAE/RMSE/MAPE per horizon — these numbers go directly into your report's results table

**Day 20**
- [ ] Wrap the trained model into a clean `predict(recent_window) → future_speeds` function
- [ ] Test it standalone with a sample input

**Day 21 — Buffer / catch-up (protect this one)**
- [ ] This is the slack that absorbs Week 3 overflow — don't schedule new work here unless training finished early

**✅ End of Week 3:** trained model checkpoint + recorded MAE/RMSE/MAPE numbers for the report.

---

## Week 4 — Route Optimization & Backend

**Day 22**
- [ ] Build the routing graph over the 207 sensors using `edge_index`/`edge_attr` from the loader
- [ ] Convert predicted speed → predicted travel time per edge (time = distance / speed, using sensor-pair distances)

**Day 23**
- [ ] Implement shortest path (`networkx.shortest_path` with predicted-time weights, or hand-rolled Dijkstra for more "built it myself" credit)
- [ ] Test on 2–3 sensor pairs — confirm sensible routes come out

**Day 24**
- [ ] Compare predicted-time route vs. static-current-time route on the same pairs — this comparison is your headline "why this matters" result

**Day 25**
- [ ] Set up FastAPI project structure
- [ ] Build `/predict` endpoint: recent traffic window → predicted congestion per sensor (JSON)

**Day 26**
- [ ] Build `/route` endpoint: origin/destination sensor IDs → runs `predict()` → runs route optimizer → returns route + ETA

**Day 27**
- [ ] End-to-end local test of both endpoints (curl/Postman)
- [ ] Add error handling (invalid sensor ID, no path found); log predictions to SQLite for the report's evaluation section

**Day 28 — Buffer / catch-up**

**✅ End of Week 4:** working `/predict` and `/route` endpoints, tested locally.

---

## Week 5 — Frontend & Integration

**Day 29**
- [ ] Set up Leaflet.js map centered on LA
- [ ] Plot the 207 sensor points using saved lat/lon

**Day 30**
- [ ] Draw sensor-to-sensor edges, color-coded by predicted congestion (green/yellow/red), pulling from `/predict`

**Day 31**
- [ ] Add origin/destination selection (dropdown or click-to-select)
- [ ] Call `/route`, draw the highlighted path, show predicted ETA vs. static-route ETA side by side

**Day 32**
- [ ] Polish UI: legend, loading states, layout
- [ ] Full click-through test of the whole flow

**Day 33**
- [ ] Fix bugs found in testing
- [ ] Pick 2–3 rehearsed demo scenarios — origin/destination pairs where the predicted route clearly beats the static one (don't leave this to chance live)

**Day 34**
- [ ] Record a backup screen-capture demo video in case the live demo has issues during viva
- [ ] Optional: deploy on a free tier (Render/Railway) so it's accessible beyond your laptop

**Day 35 — Buffer / catch-up**

**✅ End of Week 5:** working end-to-end demo (map + prediction + route) with 2–3 rehearsed scenarios and a backup video.

---

## Week 6 — Report & Viva Prep

**Day 36**
- [ ] Compile results into tables/charts: MAE/RMSE/MAPE per horizon, route time-savings %
- [ ] Draft report skeleton (Abstract, Intro, Literature Survey, Methodology, Implementation, Results, Conclusion)

**Day 37**
- [ ] Write Methodology + Implementation sections while the details are fresh
- [ ] Insert the architecture diagram

**Day 38**
- [ ] Write Results + Evaluation with your real numbers and a benchmark comparison table
- [ ] Write Conclusion + Future Work — frame full street-level routing, PeMS-BAY cross-validation, and live simulation as future work (this turns cut scope into an accepted framing, not a gap)

**Day 39**
- [ ] Build presentation slides (10–15 slides): problem, architecture, demo screenshots, results, conclusion
- [ ] Proofread the report

**Day 40**
- [ ] Rehearse viva out loud using your demo scenarios
- [ ] Prepare answers for likely questions (why GNN? why METR-LA? what's your MAE? what would you improve with more time?)

**Day 41**
- [ ] Second, timed rehearsal — in front of someone if possible
- [ ] Final formatting pass on the report (headings, citations, page numbers)

**Day 42 — Buffer / final checks**
- [ ] Submit report, test the demo on the actual presentation machine, charge devices

**✅ End of Week 6:** finished report, slides, and a rehearsed viva.

---

## If You Only Have 4–5 Weeks

Cut in this order — this preserves the core pipeline (data → model → routing → API → demo), which is what actually gets evaluated:

1. Drop every buffer day except Day 21 (Week 3's) — saves 5 days. Training is the one place you can't safely skip slack.
2. Skip Day 34's deployment step — a local demo is enough.
3. Skip Day 18's hyperparameter tuning if training converges reasonably on the first pass.
4. Merge Week 6 into 4 days — write Methodology and Results in the same sitting, do one rehearsal instead of two.

This gets you to roughly 30–32 days without touching the technical core.

---

## Quick Reference

- **Core libraries:** `torch`, `torch-geometric`, `torch-geometric-temporal`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `networkx`
- **Dataset:** METR-LA via `torch_geometric_temporal.dataset.METRLADatasetLoader` — 207 sensors, LA highways, 5-min intervals, March–June 2012
- **Reference implementation:** `a3tgcn_for_traffic_forecasting.ipynb` in the `pytorch_geometric_temporal` GitHub repo
- **Sensor coordinates:** `graph_sensor_locations.csv`, from the original DCRNN repo on GitHub (search "DCRNN traffic github")
- **Benchmarks to cite:** DCRNN / T-GCN / A3TGCN papers' MAE/RMSE/MAPE tables on METR-LA
