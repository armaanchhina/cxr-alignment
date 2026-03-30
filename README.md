# Multimodal Chest X-ray Retrieval

This project trains a multimodal model to align chest X-ray images with radiology reports using contrastive learning. The goal is to retrieve the correct report given an image and vice versa, and evaluate performance using Recall@K.

## Setup

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
- Train the multimodal model
- Save the best checkpoint
- Log metrics per epoch

Outputs will be saved to:

```
outputs/
  checkpoints/
    best_model.pt
  results/
    metrics_epoch_*.json
```

## Run Evaluation

After training:

```bash
python eval.py
```

This will:
- Load the saved checkpoint
- Evaluate retrieval performance
- Save final metrics to `outputs/results/final_metrics.json`

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
- `cxr-align.json`
- `IMAGE_FILENAMES`
- Downloaded images inside `images/`

## Notes

- A fixed random seed is used for the train/validation split
- GPU is recommended for training
- If full dataset reproduction is not possible, a smaller subset can be used
- Make sure credentials and paths are set correctly before running

## Full Workflow

```bash
export PHYSIONET_USERNAME=your_username
python scripts/download_images.py
python train.py
python eval.py
```

This will reproduce the full training and evaluation pipeline.