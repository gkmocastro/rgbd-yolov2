import torch
from torch.utils.data import DataLoader
from dataset import YoloDarknetDataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from model import model_builder
import lightnet as ln


def train_yolov2(model, 
                 train_dataloader, 
                 loss_fn, 
                 optimizer, 
                 num_epochs, 
                 device,
                 data_mod="rgbd"):
    """
    Trains the YOLOv2 model.

    Args:
        model (torch.nn.Module): YOLOv2 model.
        dataset (torch.utils.data.Dataset): Custom dataset providing (image, targets) pairs.
        loss_fn (function): Loss function for YOLOv2.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        list of float: Average loss per epoch.
    """
    # Move the model to the specified device
    model.to(device)
    
    # Store loss per epoch
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Track total loss for the epoch

        for batch, (X, y) in enumerate(train_dataloader):
            # images = torch.stack([item[0] for item in batch]).to(device)  # Stack images in the batch and send to device
            # targets = torch.stack([item[1] for item in batch]).to(device)  # Stack labels in the batch and send to device
            images = X.to(device)
            targets = y["boxes"]
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = loss_fn(outputs.cpu(), targets.cpu())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate and store average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if epoch == 0:
            best_loss = avg_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
            if epoch > 60:
                model.save(f"models/{data_mod}_{epoch}.pth")
                print(f"Saving new best model Epoch: {epoch} | Loss: {avg_loss}")

        with open("loss_rgbd_anything.txt", "w") as file:
            file.write(f"{avg_loss}\n")

        epoch_losses.append(avg_loss)
        

    return None

if __name__=="__main__":

    #IMG_DIR = "/home/gustavo/workstation/depth_estimation/codes/rgbd-yolov2/data/images_test/"
    TRAIN_IMG_DIR = "/home/escorpiao/workspace/depth-anything-dataset/Depth-Anything-V2/data/train/images"
    TRAIN_DEPTH_DIR = "/home/escorpiao/workspace/depth-anything-dataset/Depth-Anything-V2/data/output_train"
    #LABEL_DIR =  "/home/gustavo/workstation/depth_estimation/codes/rgbd-yolov2/data/labels"
    TRAIN_LABEL_DIR =  "/home/escorpiao/workspace/depth-anything-dataset/Depth-Anything-V2/data/train/labels"
    BATCH_SIZE = 16
    NUM_WORKERS = 32
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = "rgbd"
    FUSE_LAYER = 15

    print(f"Using Device {DEVICE}")

    model = model_builder(num_classes=3, model_type=MODEL_TYPE, fuse_layer=FUSE_LAYER).to(DEVICE)

    loss_fn = ln.network.loss.RegionLoss(
        num_classes= model.num_classes,
        anchors=model.anchors,
        network_stride=model.stride
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_transforms = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    train_dataset = YoloDarknetDataset(
        images_dir=TRAIN_IMG_DIR,
        depth_dir=TRAIN_DEPTH_DIR,
        labels_dir=TRAIN_LABEL_DIR,
        classes=["Cyclist", "Pedestrian", "Car"],
        transform=train_transforms,
        model_type=MODEL_TYPE
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    train_yolov2(model=model, 
                train_dataloader=train_dataloader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                num_epochs=NUM_EPOCHS, 
                device=DEVICE,
                data_mod=MODEL_TYPE)