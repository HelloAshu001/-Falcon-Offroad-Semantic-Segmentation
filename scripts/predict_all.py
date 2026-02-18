import torch
import os
import yaml
import cv2
import numpy as np

from models.deeplabv3plus import get_model

cfg = yaml.safe_load(open("configs/config.yaml"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = get_model(cfg["model"]["num_classes"])
model.load_state_dict(
    torch.load(
        os.path.join(cfg["output"]["checkpoint_dir"], "best_model.pth"),
        map_location=device
    )
)

model.to(device)
model.eval()

os.makedirs(cfg["output"]["results_dir"], exist_ok=True)

TARGET_HEIGHT = 544
TARGET_WIDTH = 960

for name in os.listdir(cfg["dataset"]["test_images"]):

    path = os.path.join(cfg["dataset"]["test_images"], name)

    print(f"Processing: {name}")

    img = cv2.imread(path)

    if img is None:
        print(f"Skipping invalid image: {name}")
        continue

    img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

    t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t)

    pred = torch.argmax(pred, dim=1).cpu().numpy()[0]

    pred_vis = (pred * 255).astype(np.uint8)


    save_path = os.path.join(cfg["output"]["results_dir"], name)
    cv2.imwrite(save_path, pred_vis)

    print(f"Saved: {save_path}")

print("\nAll predictions completed.")

