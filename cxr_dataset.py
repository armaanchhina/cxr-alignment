import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CXRMultimodalDataset(Dataset):
    def __init__(self, samples, tokenizer, transform=None, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        for item in samples:
            report_id = item["id"]
            image_path = os.path.join("images", f"{report_id}.jpg")

            if os.path.exists(image_path):
                self.samples.append({
                    "id": report_id,
                    "report": item["report"],
                    "image_path": image_path
                })
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image_path"]).convert("RGB")

        encode = self.tokenizer(
            item["report"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "images": image,
            "input_ids": encode["input_ids"].squeeze(0),
            "attention_mask": encode["attention_mask"].squeeze(0),
            "report_id": item["id"]
        }