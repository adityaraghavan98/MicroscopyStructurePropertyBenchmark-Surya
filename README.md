# MicroscopyStructurePropertyBenchmark - In active development  - This Readme might not be updated!

**What it is** — Minimal active-learning loop for structure-property relationships in electron microscopy. Small, modular, easy to extend.

**Pipeline** — `dataset → representation → model → acquisition → reward/metric → log`

---

**Quick start**

​```bash
uv sync --extra dev

uv run mspb --config configs/pca_gp_ei.yaml
​```

Real STEM file: place `test_stem.h5` at `data/raw/`, use `configs/stem_pca_gp_ei.yaml`.  
Sweep all methods: `uv run mspb-sweep --config configs/stem_all_methods.yaml`

**Plotting results? - only available for mspb-sweep (i.e all methods sweep option)**
```python
Plotting: 
uv run python scripts/postprocess_csv.py --csv outputs/stem_all_methods_100_steps.csv
```
---

**Datasets** — `synthetic`, `stem_h5`

**Representations** — `pca`, `patches`

**Models** — `gpytorch_gp`, `dkl`

**Acquisition** — `EI`, `UCB`, `beacon`, `random`

**Rewards** — `dipole` (0.35–0.55 eV), `edge` (0.60–0.75 eV), `bulk` (0.80–1.00 eV), `zero`

---

**Outputs per run**

- `predictions_BO_step<N>.png/.pkl` — mean, variance, true scalarizer per step
- `Active_learning_statistics.pkl` — acquisition order, seeds, traces, coords
- `AL_traj.png` — trajectory over image
- `run.log` + `training_log.jsonl` — human and structured logs
- `checkpoints/model_step<N>.pt` + `latest.pt` (optional)
- Sweep CSV: one row per method per step — `mse, mae, nlpd, coverage, loss`

---

**Adding a method** — implement `representation`, `model.fit/predict`, `acquisition`, `runner`. Add a config in `configs/` and a smoke test in `tests/`.

---

