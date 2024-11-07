import torch
from collections import defaultdict
import numpy as np

def calculate_iou(box1, box2):
    # Calculate Intersection over Union (IoU) for two bounding boxes
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def compute_ap(precision, recall):
    # Calculate Average Precision (AP) using precision and recall values
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def calculate_map(yolo_model, test_dataloader, nms_function, device, iou_threshold=0.5, confidence_threshold=0.5):
    # Store true positives, scores, and number of ground truths per class
    true_positives = defaultdict(list)
    scores = defaultdict(list)
    num_ground_truths = defaultdict(int)


    yolo_model.to(device)
    yolo_model.eval()


    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            outputs = yolo_model(images)
            
            for i, output in enumerate(outputs):
                # Apply NMS and get predictions above confidence threshold
                predictions = nms_function(output)
                ground_truths = targets[i]  # Ground truth boxes for the image
                
                # Count ground truth boxes for each class
                for gt in ground_truths:
                    num_ground_truths[int(gt[4])] += 1  # gt[4] is class ID
                    
                detected = []
                for pred_box in predictions:
                    pred_class = int(pred_box[5])
                    pred_score = pred_box[4]
                    scores[pred_class].append(pred_score)
                    
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(ground_truths):
                        if int(gt[4]) == pred_class and gt_idx not in detected:
                            iou = calculate_iou(pred_box[:4], gt[:4])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    if best_iou >= iou_threshold:
                        true_positives[pred_class].append(1)
                        detected.append(best_gt_idx)
                    else:
                        true_positives[pred_class].append(0)

    # Calculate precision, recall, and average precision for each class
    average_precisions = {}
    for c in num_ground_truths.keys():
        if num_ground_truths[c] == 0:
            continue
        
        sorted_indices = np.argsort(-np.array(scores[c]))
        true_positives_sorted = np.array(true_positives[c])[sorted_indices]
        
        tp_cumsum = np.cumsum(true_positives_sorted)
        fp_cumsum = np.cumsum(1 - true_positives_sorted)
        
        recall = tp_cumsum / num_ground_truths[c]
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        average_precisions[c] = compute_ap(precision, recall)
    
    # Compute mAP as the mean of average precisions for all classes
    mean_ap = np.mean(list(average_precisions.values()))
    return average_precisions, mean_ap

# Usage example
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# average_precisions, mAP = calculate_map(yolo_model, test_dataloader, nms_function)
# print("Average Precisions per Class:", average_precisions)
# print("Mean Average Precision (mAP):", mAP)
