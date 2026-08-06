# LandslideNet

LandslideNet is a Python framework for large-area landslide susceptibility
mapping from aligned raster factors and a landslide inventory. It provides a
complete workflow for terrain-based regionalization, fold-wise model training,
and susceptibility-map generation.

## Main capabilities

- Build continuous physiographic macro-regions from terrain factors.
- Prepare fold-specific raster features and positive/negative samples.
- Train LandslideNet, classical machine-learning models, and controlled deep
  learning variants.
- Compare DWSS and random negative-sampling strategies.
- Produce full-domain ensemble susceptibility maps.
- Save fold metrics, predictions, training histories, and experiment metadata.

## Repository structure

```text
.
├── 1_data_processing.py              # Build or check macro-regions
├── 2_model_train.py                  # Train regional experiments
├── 3_model_predict.py                # Generate susceptibility maps
├── Landslide_susceptibility_mapping.xml  # Main project configuration
├── models/                           # LandslideNet and all comparison models
├── utils/                            # Data, training, prediction, and reporting code
├── environment.yml                   # Conda environment
├── pyproject.toml                    # Python package metadata
└── LICENSE
```

## Requirements

- Linux or another rasterio-compatible operating system
- Python 3.10 or newer
- PyTorch, rasterio, NumPy, SciPy, pandas, scikit-learn, and related packages

Create the supplied Conda environment with:

```bash
conda env create -f environment.yml
conda activate landslidenet
```

If the environment already exists, update it with:

```bash
conda env update -n landslidenet -f environment.yml
```

## Input data and configuration

Edit `Landslide_susceptibility_mapping.xml` before running the pipeline. The
configuration specifies:

- the directory containing aligned factor rasters;
- the point-vector landslide inventory (or a legacy aligned raster);
- the number and output path of macro-regions;
- regional split and sampling parameters;
- model selection and training hyperparameters;
- training and prediction output directories.

Factor rasters must share the same grid, extent, resolution, and coordinate
reference system. Point inventories are reprojected to this grid and duplicate
points in the same cell are removed. Paths in the XML may be absolute or
changed to match a local data layout.

DWSS is fitted independently inside every outer fold from inner-training
regions only. It uses the joint multivariate Gaussian KDE, `theta_min = 0.55`,
three Jenks natural-break strata, a 1:1 class ratio, and the manuscript
allocation based on each stratum's mean divergence. If an initial uniform
candidate pool cannot meet a stratum quota, the trainer draws additional
uniform background proposals and safely prunes impossible candidates with a
Gaussian-kernel upper bound before exact KDE evaluation. Quotas are never
silently moved between strata; all capacities and selected counts are recorded
in `fold_leakage_audit.json` and `sampling_diagnostics.json`.

The default training controls are designed for the strongly unequal support of
the frozen continuous regions without reading the outer test block. Regional
sample weights are normalized separately inside each class, so the required
1:1 positive/negative loss prior is preserved. Deep models use the same
source-region V-REx penalty, Aspect-aware flip/rotation augmentation, five-epoch
learning-rate warm-up, validation-AUC checkpoint selection, plateau-based
learning-rate reduction, and a 60-epoch minimum before early stopping. The
classification threshold is selected only on the validation region. It is
searched at the exact validation scores and, within the configured F1
tolerance, favors the smallest precision-recall gap. These controls are shared
by both DWSS/random arms and by every applicable comparison model.

## Quick start

Run the three stages in order:

```bash
cd /home/whu/桌面/myCode/LandslideNet

# 1. Build or check terrain-derived macro-regions.
conda run -n landslidenet python 1_data_processing.py \
  Landslide_susceptibility_mapping.xml

# 2. Train the configured model(s).
conda run -n landslidenet python 2_model_train.py \
  Landslide_susceptibility_mapping.xml \
  --models landslidenet \
  --output-dir /home/whu/桌面/myResult/LandslideNet/02_experiment

# 3. Generate full-domain susceptibility maps.
conda run -n landslidenet python 3_model_predict.py \
  /home/whu/桌面/myResult/LandslideNet/02_experiment \
  --models landslidenet \
  --sampling-methods dwss random
```

To continue an interrupted training run, repeat the training command with
`--resume`. Use `--overwrite` only when existing outputs should be replaced.
Because the resolved training protocol is part of the resume contract, results
created with an older XML snapshot cannot be resumed with the revised controls;
start a new output directory instead.

## Model selection

The model can be selected in the XML file or overridden from the command line.
The following names and groups are available:

```text
landslidenet
machine_learning
deep_learning
ablation
proposed
all
```

Individual registered models include `logistic_regression`, `catboost`,
`extra_trees`, `lightgbm`, `random_forest`, `baseline`, `only_dcse`,
`only_spb`, `dbpfnet`, `da_lsf`, and `lgc_net`.

For example:

```bash
conda run -n landslidenet python 2_model_train.py \
  Landslide_susceptibility_mapping.xml \
  --models proposed machine_learning
```

## Prediction modes

`3_model_predict.py` combines the trained fold models into a full-domain
susceptibility map. Deep models use multi-stride median fusion; classical
models use direct per-pixel inference.

Use `--sampling-methods dwss random` to generate maps for both sampling
strategies, or specify only one method.

## Outputs

Training and prediction results are written below the selected output
directory. Typical files include:

- fold-specific model checkpoints and configuration snapshots;
- training history and evaluation metric tables;
- `all_models_5fold_metrics.csv`, containing fold rows plus five-fold
  mean/standard-deviation rows for every model and sampling arm;
- `all_models_training_history.csv`, containing every available training or
  boosting round across folds, models, and sampling arms;
- raster susceptibility maps and optional binary maps;
- sample manifests and regional split information;
- JSON metadata describing the completed experiment.

## License

See [LICENSE](LICENSE).
