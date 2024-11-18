import torch
import lightnet as ln
from utils import to_pixel_coords


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device):
    # Put model in eval mode
    model.eval() 

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


            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            # NEED TO ADAPT TO THE BATCH OF IMAGES
            
            for box in y:
                if box[0] != -1: #supress the padding
                    x1, y1, x2, y2 = to_pixel_coords(box[1:], 416, 416)
                    # add image index on the list
                    pred_box = [img_index, box[0].item(), 1, x1.item(), y1.item(), x2.item(), y2.item()] 
                    true_boxes.append(pred_box)

            for bbox in output_boxes:
                x, y, h, w = to_pixel_coords(bbox[1:5], 1, 1)
                pred_boxes.append(
                    [img_index, 
                    bbox[6].item(), 
                    bbox[5].item() , 
                    x.item(), 
                    y.item(),
                    h.item(), 
                    w.item()])


            img_index += 1
            
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    # test_acc = test_acc / len(dataloader)
    return test_loss, pred_boxes, true_boxes