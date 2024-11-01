import torch
from torch.utils.data import DataLoader
from dataset import YoloDarknetDataset
from torchvision import transforms
import os


def train_yolov2(model, train_dataloader, loss_fn, optimizer, num_epochs, device):
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
            targets = y["boxes"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate and store average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses

if __name__=="__main__":

    IMG_DIR = "data/data_object_image_2/training/image_2"
    LABEL_DIR =  "data/labels"
    BATCH_SIZE = 8
    NUM_WORKERS = os.cpu_count()


    train_transforms = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])


    train_dataset = YoloDarknetDataset(
        images_dir=IMG_DIR,
        labels_dir=LABEL_DIR,
        classes=["Cyclist", "Pedestrian", "car"],
        transform=train_transforms,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    train_yolov2()