import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import draw_bounding_boxes, to_xyxy_coords


def debug_training_sample(model, dataloader, loss_fn, device, img_size=(416, 416), sample_index=0):
    """
    Debugs the training by displaying a single sample from the dataloader, showing true and predicted boxes (visually and as text), and displaying the loss value.
    Args:
        model: The model to debug.
        dataloader: The dataloader to sample from.
        loss_fn: The loss function.
        device: The device to use.
        img_size: Tuple of (width, height) for image size.
        sample_index: Index of the sample to debug (default: 0).
    """
    model.eval()
    model.to(device)

    # Get a single batch (first sample only)
    X, y = next(iter(dataloader))
    image = X[sample_index].to(device)  # (C, H, W)
    target_boxes = y["boxes"][sample_index]  # (num_boxes, 5) or similar

    # Forward pass (no grad)
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Compute loss (move to CPU for loss_fn if needed)
    loss = loss_fn(output.cpu(), y["boxes"][sample_index:sample_index+1].cpu())

    # Prepare true boxes for visualization and text
    true_boxes = []
    for box in target_boxes:
        if box[0].item() == -1:
            continue
        x1, y1, x2, y2 = to_xyxy_coords(box[1:], img_size[0], img_size[1])
        true_boxes.append([0, int(box[0].item()), 1, x1.item(), y1.item(), x2.item(), y2.item()])

    # Prepare predicted boxes (assuming model output is compatible)
    # You may need to adapt this part depending on your model's output format
    pred_boxes = []
    if hasattr(model, 'anchors') and hasattr(model, 'stride'):
        import lightnet as ln
        GetBoxes_fn = ln.data.transform.GetAnchorBoxes(
            conf_thresh=0.5,
            network_stride=model.stride,
            anchors=model.anchors
        )
        nms_fn = ln.data.transform.NMS(iou_thresh=0.5, class_nms=True)
        output_boxes = GetBoxes_fn(output)
        output_boxes = nms_fn(output_boxes)
        for bbox in output_boxes:
            x1, y1, x2, y2 = to_xyxy_coords(bbox[1:5], img_size[0]/416, img_size[1]/416)
            pred_boxes.append([
                0,
                int(bbox[6].item()),
                float(bbox[5].item()),
                x1.item(),
                y1.item(),
                x2.item(),
                y2.item()
            ])
    else:
        print("Model does not have anchors/stride attributes. Skipping predicted box visualization.")

    # Convert image tensor to numpy for visualization
    image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    elif image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]

    # Draw boxes
    image_with_boxes = draw_bounding_boxes(image_np, pred_boxes, true_boxes)

    # Show image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.title("True (green) and Predicted (blue) Boxes")
    plt.show()

    # Print true and predicted boxes as text
    print("True Boxes:")
    for box in true_boxes:
        print(box)
    print("\nPredicted Boxes:")
    for box in pred_boxes:
        print(box)
    print(f"\nLoss value for this sample: {loss.item():.4f}")


def debug_training_sample_extra(model, dataloader, loss_fn, device, img_size=(416, 416), sample_index=0, batch_index=0):
    """
    Debugs the training by displaying a single sample from the dataloader, showing true and predicted boxes (visually and as text), and displaying the loss value.
    Args:
        model: The model to debug.
        dataloader: The dataloader to sample from.
        loss_fn: The loss function.
        device: The device to use.
        img_size: Tuple of (width, height) for image size.
        sample_index: Index of the sample to debug (default: 0).
    """
    model.eval()
    model.to(device)

    iter_dataloader = iter(dataloader)

    # Get batch_index batch
    for i in range(batch_index+1):
        X, y = next(iter_dataloader)

    image = X[sample_index].to(device)  # (C, H, W)
    target_boxes = y["boxes"][sample_index]  # (num_boxes, 5) or similar

    # Forward pass (no grad)
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Compute loss (move to CPU for loss_fn if needed)
    loss = loss_fn(output.cpu(), y["boxes"][sample_index:sample_index+1].cpu())

    # Prepare true boxes for visualization and text
    true_boxes = []
    for box in target_boxes:
        if box[0].item() == -1:
            continue
        x1, y1, x2, y2 = to_xyxy_coords(box[1:], img_size[0], img_size[1])
        true_boxes.append([0, int(box[0].item()), 1, x1.item(), y1.item(), x2.item(), y2.item()])

    # Prepare predicted boxes (assuming model output is compatible)
    # You may need to adapt this part depending on your model's output format
    pred_boxes = []
    if hasattr(model, 'anchors') and hasattr(model, 'stride'):
        import lightnet as ln
        GetBoxes_fn = ln.data.transform.GetAnchorBoxes(
            conf_thresh=0.5,
            network_stride=model.stride,
            anchors=model.anchors
        )
        nms_fn = ln.data.transform.NMS(iou_thresh=0.5, class_nms=True)
        output_boxes = GetBoxes_fn(output)
        output_boxes = nms_fn(output_boxes)
        for bbox in output_boxes:
            x1, y1, x2, y2 = to_xyxy_coords(bbox[1:5], img_size[0]/416, img_size[1]/416)
            pred_boxes.append([
                0,
                int(bbox[6].item()),
                float(bbox[5].item()),
                x1.item(),
                y1.item(),
                x2.item(),
                y2.item()
            ])
    else:
        print("Model does not have anchors/stride attributes. Skipping predicted box visualization.")

    # Convert image tensor to numpy for visualization
    image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    elif image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]

    # Draw boxes
    image_with_boxes = draw_bounding_boxes(image_np, pred_boxes, true_boxes)

    # Show image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.title("True (green) and Predicted (blue) Boxes")
    plt.show()

    # Print true and predicted boxes as text
    print("True Boxes:")
    for box in true_boxes:
        print(box)
    print("\nPredicted Boxes:")
    for box in pred_boxes:
        print(box)
    print(f"\nLoss value for this sample: {loss.item():.4f}") 