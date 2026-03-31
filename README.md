# Multimodal Chest X-ray Retrieval

This project trains a multimodal model to align chest X-ray images with radiology reports using finding-aware contrastive learning. The goal is to retrieve the correct report given an image (image-to-text) and vice versa (text-to-image), evaluated using Recall@K.

The model encodes images with DenseNet-121 and reports with Bio_ClinicalBERT, projecting both into a shared 512-dimensional L2-normalized embedding space. The contrastive loss groups positives by shared radiological finding, encouraging the model to learn clinically meaningful alignment.

## Setup

1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

This project uses data from PhysioNet. You must have approved access before running anything.

Dataset link:  
https://physionet.org/content/cxr-align/1.0.0/

IMAGE_FILENAMES link:
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

You may also need access to MIMIC-CXR-JPG for downloading images.

After you have access, export your PhysioNet username:

```bash
export PHYSIONET_USERNAME=your_username
```

You will be prompted for your password when downloading images.

## Download Images

Before training, download the required chest X-ray images:

```bash
python scripts/download_images.py
```

This script will:
- Build the image ID to path mapping
- Generate required image URLs
- Download images into the `images/` folder

## Train the Model

```bash
python train.py
```

This will:
- Load paired image and report data
- Train the multimodal model using finding-aware contrastive loss
- Save the best checkpoint
- Log metrics per epoch

Outputs will be saved to:

```
outputs/
  best_model.pt
  metrics_epoch_*.json
```

> **Note:** Training was run overnight (~8–12 hours). GPU is strongly recommended; CPU training is not practical for the full dataset.

## Run Evaluation

After training:

```bash
python eval.py
```

This will:
- Load the saved checkpoint from `outputs/checkpoints/best_model.pt`
- Evaluate retrieval performance
- Save final metrics to `outputs/results/final_metrics.json`

## Results

Evaluation on the full dataset using Recall@K:

| Metric | @1 | @5 | @10 |
|---|---|---|---|
| Image → Text (exact) | 0.0058 | 0.0291 | 0.0552 |
| Text → Image (exact) | 0.0145 | 0.0262 | 0.0523 |
| Image → Text (finding) | 0.1512 | 0.2267 | 0.2791 |
| Text → Image (finding) | 0.5494 | 0.6366 | 0.8110 |

**Exact** recall measures retrieval of the specific paired report/image. **Finding** recall measures retrieval of any report/image sharing the same radiological finding — the primary metric given the finding-aware training objective. The strong finding-level recall (T2I R@10: 0.811) reflects that the model has learned clinically meaningful alignment even when exact retrieval is difficult.

## Project Structure

```
.
├── train.py
├── eval.py
├── src/
│   ├── models/
│   │   └── multimodal_cxr.py
│   ├── data/
│   │   └── cxr_dataset.py
│   └── training/
│       └── train_utils.py
├── scripts/
│   └── download_images.py
├── outputs/
├── images/
├── cxr-align.json
└── IMAGE_FILENAMES
```

## Expected Files

Before training, make sure you have:
- `cxr-align.json` — paired image/report data with finding labels
- `IMAGE_FILENAMES` — mapping of image IDs to file paths
- Downloaded images inside `images/`

> **Note:** `cxr-align.json` and `IMAGE_FILENAMES` paths are currently hardcoded in the data loading scripts. If your directory layout differs, update the paths in `src/data/cxr_dataset.py` accordingly.

## Notes

- A fixed random seed is used for the train/validation split to ensure reproducibility
- GPU is required for practical training; overnight runtime should be expected on the full dataset
- If full dataset reproduction is not possible, a smaller subset can be used by modifying the dataset loader

## Full Workflow

```bash
export PHYSIONET_USERNAME=your_username
python scripts/download_images.py
python train.py
python eval.py
```

This will reproduce the full training and evaluation pipeline.