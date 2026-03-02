from torch.optim import AdamW
import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits,labels)

        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

        if step % 50 == 0:
            print(f"step {step}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / len(loader)
    
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct/total

@torch.no_grad()
def evaluate_metrics(model, loader, device, num_classes=3):
    model.eval()

    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        for t, p in zip(labels, preds):
            conf[t.long(), p.long()] += 1

    acc = conf.diag().sum().item() / conf.sum().item()

    f1s = []
    precisions = []
    recalls = []

    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp+fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec+rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    macro_f1 = sum(f1s) / num_classes
    return acc, macro_f1, conf, precisions, recalls, f1s
