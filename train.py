import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt

from models.deeplabv3plus import get_model
from utils.dataset import FalconDataset

def get_iou(preds, masks, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * masks).sum()
    union = (preds + masks).sum() - intersection
    return (intersection + smooth) / (union + smooth)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        if_flat = inputs.view(-1)
        tf_flat = targets.view(-1)
        
        intersection = (if_flat * tf_flat).sum()
        dice = (2. * intersection + smooth) / (if_flat.sum() + tf_flat.sum() + smooth)
        bce = nn.functional.binary_cross_entropy(if_flat, tf_flat, reduction='mean')
        return bce + (1 - dice)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {str(device).upper()}")

    EPOCHS = 20
    BATCH_SIZE = 4
    LR = 1e-4

    model = get_model(num_classes=1).to(device)

    train_ds = FalconDataset(image_dir="data/train/images", mask_dir="data/train/masks")
    val_ds = FalconDataset(image_dir="data/val/images", mask_dir="data/val/masks")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = DiceBCELoss()
    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_iou = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_iou = get_iou(outputs, masks).item()
            train_loss += loss.item()
            train_iou += batch_iou

            loop.set_postfix(loss=f"{loss.item():.4f}", batch_iou=f"{batch_iou:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        model.eval()
        val_iou_total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_iou_total += get_iou(outputs, masks).item()

        avg_val_iou = val_iou_total / len(val_loader)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("New Best Model Saved!")

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
