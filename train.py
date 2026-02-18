import os
import cv2
import yaml
import json
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp



print("Loading config...")

cfg = yaml.safe_load(open("configs/config.yaml"))

TRAIN_IMAGES = cfg["dataset"]["train_images"]
TRAIN_MASKS  = cfg["dataset"]["train_masks"]

VAL_IMAGES = cfg["dataset"]["val_images"]
VAL_MASKS  = cfg["dataset"]["val_masks"]

CHECKPOINT_DIR = cfg["output"]["checkpoint_dir"]

EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]
LR = cfg["training"]["learning_rate"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("runs", exist_ok=True)




if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)




class DesertDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))


    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):

        name = self.images[idx]

        image_path = os.path.join(self.image_dir, name)
        mask_path  = os.path.join(self.mask_dir, name)

        image = cv2.imread(image_path)
        mask  = cv2.imread(mask_path, 0)

       
        mask = (mask > 0).astype(np.float32)

       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

    

        h, w = image.shape[:2]

        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        image = cv2.copyMakeBorder(
            image,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_CONSTANT,
            value=0
        )

        mask = cv2.copyMakeBorder(
            mask,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_CONSTANT,
            value=0
        )

        image = torch.tensor(image).permute(2,0,1).float()
        mask  = torch.tensor(mask).unsqueeze(0).float()

        return image, mask



train_dataset = DesertDataset(TRAIN_IMAGES, TRAIN_MASKS)
val_dataset   = DesertDataset(VAL_IMAGES, VAL_MASKS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))




model = smp.DeepLabV3Plus(

    encoder_name="resnet34",
    encoder_weights="imagenet",

    in_channels=3,
    classes=1

)

model.to(device)




criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(

    model.parameters(),
    lr=LR

)


# -----------------------------
# IOU FUNCTION
# -----------------------------

def calculate_iou(pred, target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()



best_iou = 0

metrics = {

    "train_loss": [],
    "train_iou": [],
    "val_iou": []

}

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    train_iou  = 0

    loop = tqdm(train_loader)

    for images, masks in loop:

        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = calculate_iou(outputs, masks)

        train_loss += loss.item()
        train_iou  += iou

        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(

            loss=loss.item(),
            iou=iou

        )


    train_loss /= len(train_loader)
    train_iou  /= len(train_loader)


    

    model.eval()

    val_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)

            iou = calculate_iou(outputs, masks)

            val_iou += iou


    val_iou /= len(val_loader)


    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train IoU:  {train_iou:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")


    metrics["train_loss"].append(train_loss)
    metrics["train_iou"].append(train_iou)
    metrics["val_iou"].append(val_iou)


   

    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(

            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "best_model.pth")

        )

        print("New Best Model Saved!")




with open("runs/metrics.json", "w") as f:

    json.dump(metrics, f)


print("\nTraining Complete!")
print("Best IoU:", best_iou)
print("Model saved at runs/checkpoints/best_model.pth")
