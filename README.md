MicroscopyStructurePropertyBenchmark - In active development - This Readme might not be updated!
What it is: Minimal active-learning loop for structure-property relationships in electron microscopy. Small, modular, easy to extend.

Pipeline : dataset → representation → model → acquisition → reward/metric → log

Quick start

uv sync --extra dev

uv run mspb --config configs/pca_gp_ei.yaml
Real STEM file: place test_stem.h5 at data/raw/, use configs/stem_pca_gp_ei.yaml.
Sweep all methods: uv run mspb-sweep --config configs/stem_all_methods.yaml

Plotting results? - only available for mspb-sweep (i.e all methods sweep option)

Plotting: 
uv run python scripts/postprocess_csv.py --csv outputs/stem_all_methods_100_steps.csv

Datasets — synthetic, stem_h5

Representations — pca, patches

Models — gpytorch_gp, dkl

Acquisition — EI, UCB, beacon, random

Rewards — dipole (0.35–0.55 eV), edge (0.60–0.75 eV), bulk (0.80–1.00 eV), composition, peak_intensity, defect, gradient, zero

For STEM H5 runs, choose a reward in the dataset config:

YAML
dataset:
  name: stem_h5
  reward: composition
  reward_energy_range: [0.35, 0.55] (#choose the energy range in YAML file)
  
Use reward_energy_range for composition and peak_intensity when targeting a known elemental or spectral window. defect and gradient can run without a chosen window because they use the full spectrum by default.

Why dipole, edge, and bulk do not have separate functions

dipole, edge, and bulk all use the same mathematical operation: select a fixed energy window and sum the spectral intensity inside it. They only differ in the selected range:

dipole -> sum intensity from 0.35 to 0.55 eV
edge   -> sum intensity from 0.60 to 0.75 eV
bulk   -> sum intensity from 0.80 to 1.00 eV

Their ranges are stored in the ENERGY_RANGES dictionary in rewards.py. The shared spectrum_sum_scalarizer() function looks up the selected reward's range, finds the matching energy-axis channels, and sums them. Separate dipole_scalarizer(), edge_scalarizer(), and bulk_scalarizer() functions would repeat the same code.

Rewards with different mathematical behavior have separate functions:
composition: sums intensity inside the energy range selected in YAML.
peak_intensity: finds the maximum intensity inside the selected energy range.
defect: scores how different each full spectrum is from the median spectrum.
gradient: scores spatial changes in the integrated spectral signal.


Outputs per run:
predictions_BO_step<N>.png/.pkl — mean, variance, true scalarizer per step
Active_learning_statistics.pkl — acquisition order, seeds, traces, coords
AL_traj.png — trajectory over image
run.log + training_log.jsonl — human and structured logs
checkpoints/model_step<N>.pt + latest.pt (optional)
Sweep CSV: one row per method per step — mse, mae, nlpd, coverage, loss


Adding a method — implement representation, model.fit/predict, acquisition, runner. Add a config in configs/ and a smoke test in tests/.

