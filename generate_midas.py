import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

model_type = "MiDaS"


midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


midas.to(device)
midas.eval()


if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


image_dir = "/home/gkmo/workspace/data/final/new_splits/train/images"
image_path = Path(image_dir)


image_files = sorted([p for p in image_path.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])
output_dir = Path("/home/gkmo/workspace/data/final/midas_small/train")

with torch.no_grad():
    for i in image_files:
        print(f"Done image: {i.name}")
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        array = prediction.cpu().numpy()
        normalized = (array - array.min()) / (array.max() - array.min())
        norm_array = (normalized * 255).astype(np.uint8)
        output_filename = output_dir / f"{i.name}"
        img = Image.fromarray(norm_array).convert("L")
        img.save(output_filename)