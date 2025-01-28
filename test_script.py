import torch
import lightnet as ln
from utils import to_pixel_coords, to_xyxy_coords, mean_average_precision


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              batch_size, 
              loss_fn: torch.nn.Module,
              device):
    # Put model in eval mode
    model.eval()
    model.to(device) 

    GetBoxes_fn = ln.data.transform.GetAnchorBoxes(
                conf_thresh=0.5,
                network_stride=model.stride,
                anchors=model.anchors
            )

    nms_fn = ln.data.transform.NMS(
        iou_thresh=.3,
        class_nms=True
    )

    # Setup initial variables
    test_loss = 0
    pred_boxes = []
    true_boxes = []
    img_index = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y["boxes"].to(device)

            # 1. Forward pass
            model_output = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(model_output.cpu(), y.cpu())
            test_loss += loss.item()
            
            box_tensor = GetBoxes_fn(model_output.cpu())
            output_boxes = nms_fn(box_tensor)

            
            for img_index, boxes in enumerate(y):
                for box in boxes:
                    if box[0].item() == -1: #supress the padding
                        continue
                    # works with xyxy coords only!
                    #x1, y1, x2, y2 = box[1:]*416
                    x1, y1, x2, y2 = to_xyxy_coords(box[1:], 416, 416)
                    # add image index on the list
                    
                    true_box = [img_index+batch*batch_size, box[0].item(), 1, x1.item(), y1.item(), x2.item(), y2.item()] 
                    true_boxes.append(true_box)

            for bbox in output_boxes:
                # works with xyxy coords only!
                x1, y1, x2, y2 = to_xyxy_coords(bbox[1:5], 1, 1)
                #x1, y1, x2, y2 = bbox[1:5]
            
                pred_box = [bbox[0].item()+batch*batch_size, 
                bbox[6].item(), 
                bbox[5].item() , 
                x1.item(), 
                y1.item(),
                x2.item(), 
                y2.item()]

                pred_boxes.append(pred_box)

            pred_boxes = sorted(pred_boxes, key=lambda x: x[0])
            
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    # test_acc = test_acc / len(dataloader)

    metric_map = mean_average_precision(pred_boxes=pred_boxes, true_boxes=true_boxes, iou_threshold=0.5, box_format="corners", num_classes=3)

    return metric_map, test_loss, pred_boxes, true_boxes