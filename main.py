import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.optim import AdamW
import os
from cxr_dataset import CXRMultimodalDataset
from model import MultimodalCXRModel
from train_utils import train_one_epoch_retrieval, evaluate_retrieval


def load_file():
    with open("cxr-align.json", "r") as f:
        raw_data = json.load(f)
    return raw_data


def build_paired_samples(data: dict, images_root="images"):
    samples = []
    cases = data["mimic"]

    for report_id, case in cases.items():
        image_path = os.path.join("images", f"{report_id}.jpg")

        if os.path.exists(image_path):
            samples.append({
                "id": report_id,
                "report": case["report"],
                "image_path": image_path
            })

    return samples


def main():
    raw_data = load_file()
    samples = build_paired_samples(raw_data)

    print("Total paired samples:", len(samples))

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train size: {len(train_samples)}")
    print(f"Val size: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = CXRMultimodalDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        transform=image_transform,
        max_length=128
    )

    val_dataset = CXRMultimodalDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        transform=image_transform,
        max_length=128
    )

    print(f"Train dataset size after image check: {len(train_dataset)}")
    print(f"Val dataset size after image check: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCXRModel().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 5

    for epoch in range(epochs):
        train_loss = train_one_epoch_retrieval(model, train_loader, optimizer, device)
        recall_at_1, recall_at_5 = evaluate_retrieval(model, val_loader, device)

        print(
            f"Epoch {epoch+1} | "
            f"Train loss: {train_loss:.4f} | "
            f"R@1: {recall_at_1:.4f} | "
            f"R@5: {recall_at_5:.4f}"
        )


main()
