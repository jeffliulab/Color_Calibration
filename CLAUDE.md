# Color Calibration Card Detection — Project Guide

## What This Project Does

Automated color calibration system: given a photo containing a color calibration card (with red/green/blue reference patches + a target patch), the system detects the card via YOLO, extracts color values, and predicts the true color of the target patch under standard lighting using a trained ML model.

**Best result:** XGBoost with Bayesian-optimised hyperparameters — **Lab Mean ΔE = 4.59** (meets commercial printing standards).

## Project Storage Topology

The project lives across **GitHub + 3 HuggingFace repos**:

```
GitHub: jeffliulab/Color_Calibration            (source code, scripts, docs)
            │
            ▼ references
HuggingFace (jeffliulab):
  ├── Dataset  card-calibration-v1-data         (raw photos + augmented + training sets)
  ├── Model    card-calibration-v1              (YOLO + XGBoost + RF inference weights)
  └── Space    card-calibration-v1              (Gradio live demo)
```

| Resource | Platform | Location |
|---|---|---|
| Code | GitHub | `jeffliulab/Color_Calibration` |
| Data | HF Dataset | `jeffliulab/card-calibration-v1-data` |
| Model | HF Model | `jeffliulab/card-calibration-v1` |
| Demo | HF Space | `jeffliulab/card-calibration-v1` |

**No DVC, no GCP, no external tools.** Run `python scripts/dataset_download.py` to fetch all data.

## Repository Layout

```
Color_Calibration/
├── src/                          # Core source code
│   ├── detect/yolo.py            # YOLO-based card & pattern detection (2-stage)
│   ├── predict/predict_rf.py     # ColorPredictionSystem — full inference pipeline
│   ├── feature_extraction/       # RGB feature extraction from cropped patterns
│   ├── train/                    # Model training scripts
│   │   ├── pre_train.py          # Data loading & feature engineering from CSV
│   │   ├── train_rf.py           # Random Forest (n_estimators=500)
│   │   ├── train_xgboost.py      # XGBoost (Bayesian-tuned hyperparams)
│   │   ├── train_linear_regression.py
│   │   └── train_mlp.py          # Small neural network
│   ├── tune/tune_xgboost.py      # Bayesian hyperparameter optimisation
│   ├── data_processing/          # Data cleaning, augmentation prep
│   ├── detect_processing/        # YOLO training data prep & augmentation
│   └── api/main.py               # Legacy FastAPI deployment (Google Cloud Run)
│
├── space/                        # ** HuggingFace Space deployment **
│   ├── app.py                    # Gradio web UI
│   ├── model_utils.py            # Model download (HF Hub), detection & prediction pipeline
│   ├── requirements.txt          # Space Python dependencies
│   ├── README.md                 # HF Space YAML metadata (sdk: gradio)
│   └── checkpoints/              # Local model cache (auto-downloaded from HF Hub)
│
├── scripts/                      # Deployment & data scripts
│   ├── deploy_space.py           # Push space/ → HF Spaces
│   ├── hf_upload.py              # Upload model files → HF Model Hub
│   ├── dataset_pack.py           # Pack data/ → tar.gz archives
│   ├── dataset_upload.py         # Upload archives → HF Dataset Hub
│   └── dataset_download.py       # Download + extract data from HF
│
├── configs/detect/               # YOLOv8 training configs
├── weights/YOLOv8/yolov8n.pt    # Base YOLOv8-nano pretrained weights
├── tests/test_generalize.py      # Generalisation test
├── notebooks/                    # Jupyter experiments (exploration, not prod)
├── docs/                         # Images and documentation assets
├── data/                         # Dataset (gitignored, via dataset_download.py)
├── Dockerfile                    # Legacy: Google Cloud Run deployment
├── setup.py                      # pip install -e .
└── README.md
```

## ML Pipeline (How It Works)

### Inference Pipeline (what the Space runs)

1. **Card detection** — YOLO model 1 (`yolo_first.pt`) finds the calibration card bounding box
2. **Pattern detection** — YOLO model 2 (`yolo_second.pt`) finds 4 patches inside the card:
   - `red_circle` (red reference), `green_triangle` (green ref), `blue_pentagon` (blue ref), `black_box` (target)
3. **Feature extraction** — Extract average RGB from center 1/3 of each patch
4. **Feature engineering** — Compute 12 features:
   - 9 delta values: `(Rp - Rs)` for each channel of each reference colour
   - 3 target values: `Cp_R, Cp_G, Cp_B` (captured target RGB)
5. **Prediction** — XGBoost or Random Forest model predicts true RGB of target
6. **Evaluation** — Compute Lab ΔE (CIE76) between captured and predicted colour

### Feature names (order matters for model input)

```
Delta_RR_red, Delta_RG_red, Delta_RB_red,
Delta_RR_green, Delta_RG_green, Delta_RB_green,
Delta_RR_blue, Delta_RG_blue, Delta_RB_blue,
Cp_R, Cp_G, Cp_B
```

### Training data

- 255 hand-collected photos, augmented to ~2,294 samples (with 9,154 patch crops)
- **Stored on HuggingFace Datasets**: `jeffliulab/card-calibration-v1-data`
- Run `python scripts/dataset_download.py` to fetch all data into `data/`
- Feature CSV: `data/features/feature_0216.csv`
- Train/test split: 70/30, random_state=42

## Models & Where They Live

### HuggingFace Model Hub: `jeffliulab/card-calibration-v1`

| Hub filename | Local source | Description |
|---|---|---|
| `yolo_first.pt` | `data/models/detect/yolo_0214_first.pt` | YOLO card detector |
| `yolo_second.pt` | `data/models/detect/yolo_0214_second.pt` | YOLO pattern detector |
| `xgboost_v1.pkl` | `data/models/xgboost/model_v1.pkl` | XGBoost calibration (best) |
| `random_forest_v1.pkl` | `data/models/random_forest/model_v1.pkl` | Random Forest calibration |

### Model performance comparison

| Model | R² | RMSE | Lab Mean ΔE | Lab Median ΔE |
|---|---|---|---|---|
| **XGBoost (tuned)** | **0.8280** | **11.76** | **4.59** | **3.61** |
| Random Forest | 0.8225 | 12.10 | 5.20 | 3.96 |
| Linear Regression | 0.7113 | 14.98 | 6.63 | 5.53 |
| MLP | 0.7068 | 14.22 | 7.39 | 6.70 |

## HuggingFace Deployment

### Prerequisites

```bash
pip install huggingface_hub gradio
# Set token (one of):
echo 'hf_xxx' > ~/.hf_token
export HF_TOKEN=hf_xxx
```

### Download dataset (for training/dev)

```bash
python scripts/dataset_download.py
```

This pulls all data from `jeffliulab/card-calibration-v1-data` and extracts into `data/`. No auth needed.

### Upload models to HF Hub

```bash
# Upload all at once
python scripts/hf_upload.py --all

# Or one at a time
python scripts/hf_upload.py --file data/models/xgboost/model_v1.pkl --dest xgboost_v1.pkl
```

### Upload dataset to HF Hub

```bash
python scripts/dataset_pack.py     # pack data/ into _dataset_archive/
python scripts/dataset_upload.py   # push archives to HF Datasets
```

### Deploy Space

```bash
python scripts/deploy_space.py --space_id jeffliulab/card-calibration-v1
```

The Space (`space/app.py`) auto-downloads models from HF Hub on first request — no need to bundle checkpoints into the Space repo.

### URLs

- **Live demo:** https://huggingface.co/spaces/jeffliulab/card-calibration-v1
- **Model Hub:** https://huggingface.co/jeffliulab/card-calibration-v1
- **Dataset Hub:** https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data
- **GitHub:** https://github.com/jeffliulab/Color_Calibration

## Key Code Paths

| Task | File |
|---|---|
| Full inference (Space) | `space/model_utils.py` → `predict_color()` |
| Gradio UI | `space/app.py` → `run_calibration()` |
| YOLO detection | `src/detect/yolo.py` → `PatternDetector` |
| Color prediction class | `src/predict/predict_rf.py` → `ColorPredictionSystem` |
| Feature engineering | `src/train/pre_train.py` → `load_preprocessed_data()` |
| Train XGBoost | `src/train/train_xgboost.py` → `train_xgboost()` |
| Train Random Forest | `src/train/train_rf.py` → `train_random_forest()` |
| Hyperparameter tuning | `src/tune/tune_xgboost.py` |
| Legacy FastAPI | `src/api/main.py` |
| Deploy to HF Space | `scripts/deploy_space.py` |
| Upload models to HF | `scripts/hf_upload.py` |
| Pack dataset | `scripts/dataset_pack.py` |
| Upload dataset to HF | `scripts/dataset_upload.py` |
| Download dataset from HF | `scripts/dataset_download.py` |

## Development Notes

- Python environment was originally 3.8 (conda `machine_learning-docker-3.8`); Space uses Python 3.10
- Models are serialised with `joblib` (`.pkl` files); YOLO weights are `.pt` files
- The `data/` directory is gitignored — all data lives on HF Datasets (`jeffliulab/card-calibration-v1-data`, ~360 MB packed). Run `python scripts/dataset_download.py` to fetch.
- `space/model_utils.py` downloads models lazily from HF Hub and caches them in `space/checkpoints/`
- The Space app converts Gradio RGB input → OpenCV BGR → pipeline → results
- Reference colours are hardcoded: R=(255,0,0), G=(0,255,0), B=(0,0,255)
- There is a known bug in `src/predict/predict_rf.py` line 158-159: blue pentagon delta uses `Gp` instead of `Bp` — this is from the original repo and was preserved; the `space/model_utils.py` version has the correct implementation
