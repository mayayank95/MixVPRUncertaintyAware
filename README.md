# MixVPRUncertaintyAware

Standalone MixVPR training and evaluation with uncertainty-aware eval tooling and repo-specific dataset layout (`database/` + `queries/` via `configs/datasets.json`), forked from [UncertaintyAwareModels](https://github.com/mayayank95/UncertaintyAwareModels).

## Layout

- **`train_mixvpr.py`** — GSV-Cities training (PyTorch Lightning, `.venv-mixvpr`)
- **`eval.py`** — VPR evaluation with ECE / uncertainty metrics (`local_venv` or main `requirements-eval.txt`)
- **`data/`** — `GSVCities*`, `mixvpr_val_dataset`, `TestDataset` (db/queries image reading)
- **`mixvpr/`** — upstream MixVPR backbone, aggregator, metric-learning losses, FAISS validation
- **`models/`** — eval-time `MixVPRModel` wrapper (`method=mixvpr`)
- **`eval_metrics/`** — retrieval, ECE, baselines, W&B logging
- **`losses/`** — `vmf_loss`, `uncertainty_utils` (vMF / Gaussian uncertainty)

## Setup

A copy of `.venv-mixvpr` from UncertaintyAwareModels may already exist in this repo (not tracked by git). Otherwise:

```bash
python3.9 -m venv .venv-mixvpr
source .venv-mixvpr/bin/activate
pip install torch torchvision  # match your CUDA
pip install -r requirements-mixvpr.txt
# or: pip install -r requirements.txt  (same pins as requirements-mixvpr.txt)

cp configs/datasets.json.example configs/datasets.json
# edit data_folder and paths

export GSV_CITIES_PATH=/path/to/gsv_cities   # training only
```

Eval (separate env recommended if using torch 2.x):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

## Usage

**Train (512-d MixVPR on GSV-Cities):**

```bash
python train_mixvpr.py --max_epochs 80 --num_workers 8 --gpu 0
```

**Frozen sanity (recalls unchanged after 1 epoch):**

```bash
python train_mixvpr.py --freeze_model --max_epochs 1 --validate_before_fit --no_checkpoint
```

**Eval (Lightning checkpoint):**

```bash
python eval.py --config configs/datasets.json \
  --datasets pitts30k,msls-val --datasets_type test \
  --method mixvpr --descriptors_dimension 512 \
  --resume_model LOGS/resnet50/.../best.ckpt \
  --ckpt_state_dict_key state_dict --model_mode basic
```

## Validation data

Pitts30k / MSLS-val use `data/mixvpr_val_dataset.py` → `TestDataset` on paths from `configs/datasets.json` (not MixVPR `.mat` files).
