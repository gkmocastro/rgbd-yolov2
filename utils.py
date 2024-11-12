import torch
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image


# Helper function to convert normalized box to pixel coordinates
def to_pixel_coords(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height
    return x_min, y_min, x_max - x_min, y_max - y_min



def draw_bounding_boxes(index, model, dataset, GetBoxes_fn, nms_fn, class_names, device="cpu"):
    """
    Draws bounding boxes on an image with predictions and ground truths.

    Args:
        index (int): Index of the image in the dataset.
        model (torch.nn.Module): Trained YOLOv2 model.
        dataset (torch.utils.data.Dataset): Dataset containing the images and labels.
        nms_fn (function): Non-Maximum Suppression function to filter model outputs.
        class_names (list of str): List of class names.
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    model.eval()
    model.to(device)

    # Load image and ground truth
    image, target = dataset[index]
    target = target["boxes"]
    image = image.to(device).unsqueeze(0)  # Add batch dimension
    ground_truth_boxes = target[:, 1:]  # Get the ground truth bounding boxes
    ground_truth_classes = target[:, 0].int()  # Get the ground truth class IDs

    # Run the model to get predictions
    with torch.inference_mode():
        predictions = model(image)

    # Apply GetBoxes to transform the model output into bouding box format
    predictions = GetBoxes_fn(predictions.cpu())

    # Apply NMS to the predictions
    predictions = nms_fn(predictions)
    
    # Convert the image back to a PIL format for visualization
    image = to_pil_image(image.squeeze(0).cpu())  # Remove batch dimension

    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    


    # Draw ground truth bounding boxes
    img_width, img_height = image.size
    for i, gt_box in enumerate(ground_truth_boxes):
        if ground_truth_classes[i] == -1:  # Ignore padded values
            continue
        x, y, w, h = to_pixel_coords(gt_box.cpu(), img_width, img_height)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 10, class_names[ground_truth_classes[i]], color="green", fontsize=12, weight="bold")

    # Draw predicted bounding boxes
    for pred in predictions:
        #img_width=1 and img_heith=1 because the predictions weren't normalized
        x, y, w, h = to_pixel_coords(pred[1:5].cpu(), img_width=1, img_height=1)
        score = pred[5].item()
        class_id = int(pred[6].item())
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 10, f"{class_names[class_id]}: {score:.2f}", color="red", fontsize=12, weight="bold")

    plt.axis("off")
    plt.show()
