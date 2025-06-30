import torch
from torch.utils.data import DataLoader
from dataset import YoloDarknetDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import model_builder
import lightnet as ln
from utils import load_config, RGBDCustomTransform, DepthCustomTransform, RGBCustomTransform
from engine import train_yolov2
import argparse
    
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument("-c", "--config", help="path to config file", required=True)
args = parser.parse_args()


config = load_config(args.config)

OUTPUT_DIR = config["output_dir"]
TRAIN_IMG_DIR = config["train_img_dir"]
TRAIN_DEPTH_DIR = config["train_depth_dir"]
TRAIN_LABEL_DIR =  config["train_label_dir"]
VAL_IMG_DIR = config["val_img_dir"]
VAL_DEPTH_DIR = config["val_depth_dir"]
VAL_LABEL_DIR =  config["val_label_dir"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
MODEL_TYPE = config["model_type"]
FUSE_LAYER = config["fuse_layer"]
EXPERIMENT_NAME = config["experiment_name"]
VAL_EVERY = config["val_every"]
NORMALIZATION = config["normalization"]
DEPTH_CHECKPOINT = config["depth_checkpoint"]

# TRAIN_IMG_DIR =  BASE_DIR + TRAIN_IMG_DIR
# TRAIN_DEPTH_DIR = BASE_DIR + TRAIN_DEPTH_DIR
# TRAIN_LABEL_DIR = BASE_DIR + TRAIN_LABEL_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device {DEVICE}")

print(f"Training with config: \n {config}")

model = model_builder(num_classes=3, model_type=MODEL_TYPE, fuse_layer=FUSE_LAYER, depth_checkpoint=DEPTH_CHECKPOINT).to(DEVICE)

loss_fn = ln.network.loss.RegionLoss(
    num_classes= model.num_classes,
    anchors=model.anchors,
    network_stride=model.stride,
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0005
)

if MODEL_TYPE == "depth":
    print("Using Depth only data transforms")
    train_transforms = DepthCustomTransform(resize_size=(416, 416), flip_prob=0.5)
    val_transforms = DepthCustomTransform(resize_size=(416, 416), flip_prob=0)
elif MODEL_TYPE == "rgb":
    print("Using RGB data transforms")
    train_transforms = RGBCustomTransform(resize_size=(416, 416), flip_prob=0.5)
    val_transforms = RGBCustomTransform(resize_size=(416, 416), flip_prob=0)
else:
    print("Using RGBD data transforms")
    train_transforms = RGBDCustomTransform(resize_size=(416, 416), flip_prob=0.5)  
    val_transforms = RGBDCustomTransform(resize_size=(416, 416), flip_prob=0)

train_dataset = YoloDarknetDataset(
    images_dir=TRAIN_IMG_DIR,
    depth_dir=TRAIN_DEPTH_DIR,
    labels_dir=TRAIN_LABEL_DIR,
    classes=["Cyclist", "Pedestrian", "Car"],
    transform=train_transforms,
    model_type=MODEL_TYPE,
    normalization=NORMALIZATION
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_dataset = YoloDarknetDataset(
    images_dir=VAL_IMG_DIR,
    depth_dir=VAL_DEPTH_DIR,
    labels_dir=VAL_LABEL_DIR,
    classes=["Cyclist", "Pedestrian", "Car"],
    transform=val_transforms,
    model_type=MODEL_TYPE,
    normalization=NORMALIZATION
)


val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


train_yolov2(model=model, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            num_epochs=NUM_EPOCHS, 
            device=DEVICE,
            experiment_name=EXPERIMENT_NAME,
            output_dir=OUTPUT_DIR,
            val_every=VAL_EVERY,
            batch_size=BATCH_SIZE)