import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from cxr_dataset import CXRTextDataset
from torch.utils.data import DataLoader
from model import ClinicalBERTClassifier
import torch
import torch.nn as nn
from torch.optim import AdamW
from train_utils import train_one_epoch, evaluate

def format_text(finding, report):
    return f"FINDING {finding} [SEP] REPORT: {report}"

def load_file():
    with open("cxr-align.json", "r") as f:
        raw_data = json.load(f)
    return raw_data


def label_data(data: dict):
    samples = []
    cases = data['mimic']
    print(len(cases.keys()))
    for case_id, case in cases.items():
        chosen = case["chosen"]

        samples.append({
            "text": format_text(chosen, case["report"]),
            "label": 0 
        })

        samples.append({
            "text": format_text(chosen, case["negation"]),
            "label": 1
        })

        samples.append({
            "text": format_text(chosen, case["omitted"]),
            "label": 2
        })

    return samples



def main():
    raw_data = load_file()
    samples = label_data(raw_data)
    
    print("Total Samples: ", len(samples))

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=[s["label"] for s in samples] # keeps the label proportions same
    )

    print(f"Train size: {len(train_samples)}")
    print(f"Val size: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_dataset = CXRTextDataset(train_samples, tokenizer, max_length=128)
    val_dataset = CXRTextDataset(val_samples, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalBERTClassifier(num_classess=3).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 1

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.4f}")



main()