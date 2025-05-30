import torch
from test_script import test_step
import json
import time
import os
from pathlib import Path

def train_yolov2(model, 
                train_dataloader, 
                val_dataloader,  
                loss_fn, 
                optimizer, 
                num_epochs, 
                device,
                output_dir,
                experiment_name,
                val_every=10,
                batch_size=16):
    """
    Trains the YOLOv2 model.

    Args:
        model (torch.nn.Module): YOLOv2 model.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
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

    print(f"Start training process with device {device}")
    print(f"Experiment Name: {experiment_name}")

    output_dir = output_dir + experiment_name
    output_dir = Path(output_dir)

    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    start_training_time = time.time()
    # Store all the logs in a dict 
    log_dict = {}

    # Store loss per epoch
    train_losses = []
    val_losses = []
    val_maps = []

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # Set model to training mode
        total_loss = 0.0  # Track total loss for the epoch
        print(f"Starting epoch {epoch + 1} of {num_epochs}")
        for batch, (X, y) in enumerate(train_dataloader):
            images = X.to(device)
            targets = y["boxes"]
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images).to(device)
            # Calculate loss
            loss = loss_fn(outputs.to("cpu"), targets.to("cpu"))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate and store average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        
        end_time = time.time()
        epoch_time = end_time - start_time

        train_losses.append(avg_loss)

        # Save training loss
        with open(output_dir / f"train_loss_{experiment_name}.txt", "a") as file:
            file.write(f"{avg_loss}\n")

        print(f"-----\nEpoch [{epoch + 1}/{num_epochs}]\nLoss: {avg_loss:.4f}\nEpoch time: {epoch_time:.2f} seconds")
        
        # Validation step every 10 epochs
        if (epoch + 1) % val_every == 0:
            metric_map, val_loss = test_step(
                                    model=model,
                                    dataloader=val_dataloader,
                                    batch_size=batch_size,
                                    loss_fn=loss_fn,
                                    device=device)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation mAP|AP {metric_map}")
            

            # Save validation loss
            with open(output_dir / f"val_loss_{experiment_name}.txt", "a") as file:
                file.write(f"{val_loss}\n")

            val_losses.append(val_loss)
            val_maps.append(metric_map)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model.save(output_dir / f"{experiment_name}_best.pth")
                print(f"Saving new best model Epoch: {epoch} | Validation Loss: {val_loss}")

            with open(output_dir / f"output_{experiment_name}.txt", "w") as file:
                file.write(f"Best epoch: {best_epoch}")

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    print(f"Total Training time: {training_time:.2f} seconds")

    log_dict["train_loss"] = train_losses
    log_dict["val_loss"] = val_losses
    log_dict["best_epoch"] = best_epoch
    log_dict["best_val_loss"] = best_val_loss
    log_dict["val_maps"] = val_maps
    log_dict["training_time"] = training_time


    with open(output_dir / f"log_dict_{experiment_name}.json", "w") as file:
        json.dump(log_dict, file)


    return None