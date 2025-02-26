import torch
from test_script import test_step
import json

def train_yolov2(model, 
                 train_dataloader, 
                 loss_fn, 
                 optimizer, 
                 num_epochs, 
                 device,
                 dataset_name,
                 val_every=10):
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
        
        print(f"-----\nEpoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

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


def train_yolov2_withval(model, 
                         train_dataloader, 
                         val_dataloader,  # Add validation dataloader
                         loss_fn, 
                         optimizer, 
                         num_epochs, 
                         device,
                         dataset_name,
                         val_every=10):
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
    
    # Store loss per epoch
    epoch_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Track total loss for the epoch

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
        
        print(f"-----\nEpoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        epoch_losses.append(avg_loss)

        # Save training loss
        with open(f"output/train_loss_{dataset_name}.txt", "a") as file:
            file.write(f"{avg_loss}\n")

        # Validation step every 10 epochs
        if (epoch + 1) % val_every == 0:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.inference_mode():
                for batch, (X, y) in enumerate(val_dataloader):
                    images = X.to(device)
                    targets = y["boxes"]
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = model(images).to(device)
                    # Calculate loss
                    loss = loss_fn(outputs.to("cpu"), targets.to("cpu"))
                    
                    # Accumulate the loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save validation loss
            with open(f"output/val_loss_{dataset_name}.txt", "a") as file:
                file.write(f"{avg_val_loss}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                model.save(f"models/{dataset_name}_best.pth")
                print(f"Saving new best model Epoch: {epoch} | Validation Loss: {avg_val_loss}")

            with open(f"output/output_{dataset_name}.txt", "w") as file:
                file.write(f"Best epoch: {best_epoch}")

    return epoch_losses



def train_yolov2_withval_map(model, 
                            train_dataloader, 
                            val_dataloader,  # Add validation dataloader
                            loss_fn, 
                            optimizer, 
                            num_epochs, 
                            device,
                            dataset_name,
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
    
    # Store all the logs in a dict 
    log_dict = {}

    # Store loss per epoch
    train_losses = []
    val_losses = []
    val_maps = []

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Track total loss for the epoch

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
        
        print(f"-----\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        # Save training loss
        with open(f"output/train_loss_{dataset_name}.txt", "a") as file:
            file.write(f"{avg_loss}\n")

        # Validation step every 10 epochs
        if (epoch + 1) % val_every == 0:
            metric_map, val_loss = test_step(
                                    model=model,
                                    dataloader=val_dataloader,
                                    batch_size=batch_size,
                                    loss_fn=loss_fn,
                                    device=device)

            print(f"mAP | AP {metric_map}")
            print(f"Validation Loss: {val_loss:.4f}")

            # Save validation loss
            with open(f"output/val_loss_{dataset_name}.txt", "a") as file:
                file.write(f"{val_loss}\n")

            val_losses.append(val_loss)
            val_maps.append(metric_map)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model.save(f"models/{dataset_name}_best.pth")
                print(f"Saving new best model Epoch: {epoch} | Validation Loss: {val_loss}")

            with open(f"output/output_{dataset_name}.txt", "w") as file:
                file.write(f"Best epoch: {best_epoch}")

    log_dict["train_loss"] = train_losses
    log_dict["val_loss"] = val_losses
    log_dict["best_epoch"] = best_epoch
    log_dict["best_val_loss"] = best_val_loss
    log_dict["val_maps"] = val_maps

    with open(f"output/log_dict_{dataset_name}.json", "w") as file:
        json.dump(log_dict, file)


    return None