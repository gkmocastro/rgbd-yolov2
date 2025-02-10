import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import model_builder
import lightnet as ln
from utils import load_config

def train_yolov2(model, 
                 train_dataloader, 
                 loss_fn, 
                 optimizer, 
                 num_epochs, 
                 device,
                 dataset_name,
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
    best_epoch = 0
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
            outputs = model(images).to(device)
            # Calculate loss
            loss = loss_fn(outputs.to("cpu"), targets.to("cpu"))
            
            
            # Backward pass and optimiSzation
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
            best_epoch = epoch
            model.save(f"models/{dataset_name}_best.pth")
            print(f"Saving new best model Epoch: {epoch} | Loss: {avg_loss}")

        with open("output/loss_rgbd_anything.txt", "a") as file:
            file.write(f"{avg_loss}\n")

        with open(f"output/output_{dataset_name}.txt", "w") as file:
            file.write(f"Best epoch: {best_epoch}")

        epoch_losses.append(avg_loss)
        

    return None