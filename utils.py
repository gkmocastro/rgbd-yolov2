import torch
from collections import defaultdict
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from collections import Counter
import lightnet as ln
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# Helper function to convert normalized box to pixel coordinates
def to_pixel_coords(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height
    return x_min, y_min, x_max - x_min, y_max - y_min

def to_xyxy_coords(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height
    return x_min, y_min, x_max, y_max

def draw_bounding_boxes(image, predicted_boxes, true_boxes):
    """
    Draws predicted and ground truth bounding boxes on an image.
    
    Parameters:
    - image: The original image (NumPy array).
    - predicted_boxes: List of lists, each containing [index, class_index, confidence, x1, y1, x2, y2].
    - true_boxes: List of lists, each containing [index, class_index, x1, y1, x2, y2].
    """
    # Make a copy of the image to avoid modifying the original
    image_copy = image.copy()
    
    # Draw predicted bounding boxes (Blue)
    for box in predicted_boxes:
        _, class_index, confidence, x1, y1, x2, y2 = box
        cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        label = f"Pred {class_index}: {confidence:.2f}"
        cv2.putText(image_copy, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw ground truth bounding boxes (Green)
    for box in true_boxes:
        _, class_index,_,  x1, y1, x2, y2 = box
        cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"GT {class_index}"
        cv2.putText(image_copy, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert BGR to RGB for displaying with Matplotlib
    #image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    
    # Show the image with bounding boxes
    # plt.figure(figsize=(10, 8))
    # plt.imshow(image_copy)
    # plt.axis("off")
    # plt.show()

    return image_copy

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=3
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls).item())
    mavp = sum(average_precisions) / len(average_precisions)
    # print(f"type mavp {type(mavp)}")
    # print(f"mavp {mavp}")
    # print(f"type mavp {type(average_precisions)}")
    # print(f"average_precisions {average_precisions}")
    return mavp, average_precisions
        
def rename_state_dict(state_dict, layer_mapping):
    renamed_state_dict = {}
    for src_layer, tgt_layer in layer_mapping.items():
        if src_layer in state_dict:
            renamed_state_dict[tgt_layer] = state_dict[src_layer]
        else:
            print(f"Warning: {src_layer} not found in the source state_dict.")
    return renamed_state_dict


def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    

def plot_result(model,
                test_dataset,
                index,
                device="cpu",):
    
    GetBoxes_fn = ln.data.transform.GetAnchorBoxes(
                conf_thresh=0.5,
                network_stride=model.stride,
                anchors=model.anchors
            )

    nms_fn = ln.data.transform.NMS(
        iou_thresh=.5,
        class_nms=True
    )

    

    model.eval()
    model.to(device)

    # Load image and ground truth
    image, target = test_dataset[index]
    bbox = target["boxes"]
    #image = image.to("cpu").unsqueeze(0)  # Add batch dimension
    # ground_truth_boxes = target[:, 1:]  # Get the ground truth bounding boxes
    # ground_truth_classes = target[:, 0].int()  # Get the ground truth class IDs

    true_boxes = []
    pred_boxes = []


    with torch.inference_mode():
        model_output = model(image.unsqueeze(0))

        output_boxes = GetBoxes_fn(model_output)
        output_boxes = nms_fn(output_boxes)

    for _, boxes in enumerate(bbox):

        if boxes[0].item() == -1: #supress the padding
            continue
        # works with xyxy coords only!
        #x1, y1, x2, y2 = box[1:]*416
        x1, y1, x2, y2 = to_xyxy_coords(boxes[1:], 1242, 375)
        # add image index on the list

        true_box = [0, boxes[0].item(), 1, x1.item(), y1.item(), x2.item(), y2.item()] 
        true_boxes.append(true_box)
            
    #         true_box = [img_index+batch*batch_size, box[0].item(), 1, x1.item(), y1.item(), x2.item(), y2.item()] 
    #         true_boxes.append(true_box)

    for bbox in output_boxes:
        # works with xyxy coords only!
        x1, y1, x2, y2 = to_xyxy_coords(bbox[1:5], 1242/416, 375/416)
        #x1, y1, x2, y2 = bbox[1:5]

        pred_box = [0, 
        bbox[6].item(), 
        bbox[5].item() , 
        x1.item(), 
        y1.item(),
        x2.item(), 
        y2.item()]

        pred_boxes.append(pred_box)

    images_dir = test_dataset.images_dir
    depth_dir = test_dataset.depth_dir

    image_files = sorted([p for p in images_dir.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])

    depth_files = sorted([p for p in depth_dir.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])

    img_path = image_files[index]
    depth_path = depth_files[index]

    rgb_image = Image.open(img_path).convert("RGB") 
    depth_image = Image.open(depth_path).convert("L")

    rgb_array = np.array(rgb_image)
    depth_array = np.array(depth_image)

    rgb_wboxes = draw_bounding_boxes(rgb_array, pred_boxes, true_boxes)
    depth_wboxes = draw_bounding_boxes(depth_array, pred_boxes, true_boxes)


    #Show the image with bounding boxes
    fig, axes = plt.subplots(2,1, figsize=(10, 6))
    axes[0].imshow(rgb_wboxes)
    axes[0].axis("off")
    axes[1].imshow(depth_wboxes, cmap="gray")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


class CustomTransform(object):
    def __init__(self, resize_size=(416, 416), flip_prob=0.5):
        self.resize = transforms.Resize(resize_size)
        self.to_tensor = transforms.ToTensor()
        self.flip_prob = flip_prob

    def __call__(self, rgb_image, depth_image, target):
        # Resize both images
        rgb_image = self.resize(rgb_image)
        depth_image = self.resize(depth_image)

        # Convert bounding boxes to tensor
        bbox = torch.tensor(target["boxes"], dtype=torch.float32)

        # Horizontal flip with a given probability
        # The flip_flag is used only for visualization purposes
        if random.random() < self.flip_prob:
            rgb_image = transforms.functional.hflip(rgb_image)
            depth_image = transforms.functional.hflip(depth_image)
            # Adjust the x coordinate of the center of the bounding box
            bbox[:, 1] = 1 - bbox[:, 1]

        # Convert both images to tensors
        rgb_tensor = self.to_tensor(rgb_image)
        depth_tensor = self.to_tensor(depth_image)

        # Concatenate RGB and depth tensors along the channel dimension
        img = torch.cat((rgb_tensor, depth_tensor), 0)

        # Update target with the transformed bounding boxes
        target["boxes"] = bbox.tolist()

        return img, target