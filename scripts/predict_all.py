import torch
import os
import yaml
import cv2
import numpy as np

from models.deeplabv3plus import get_model



cfg = yaml.safe_load(open("configs/config.yaml"))

TEST_DIR = cfg["dataset"]["test_images"]
RESULT_DIR = cfg["output"]["results_dir"]
MODEL_PATH = os.path.join(cfg["output"]["checkpoint_dir"], "best_model.pth")

os.makedirs(RESULT_DIR, exist_ok=True)



if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


model = get_model(cfg["model"]["num_classes"])

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()



print("Starting predictions...")

for name in sorted(os.listdir(TEST_DIR)):

    path = os.path.join(TEST_DIR, name)

    image = cv2.imread(path)

    if image is None:
        continue

    original = image.copy()

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



    tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(device)


    with torch.no_grad():

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred.cpu().numpy()[0][0]



    pred = pred[:h, :w]



    mask = (pred > 0.5).astype(np.uint8) * 255


    save_path = os.path.join(RESULT_DIR, name)

    cv2.imwrite(save_path, mask)


print("\nAll predictions completed!")
print("Results saved in:", RESULT_DIR)
