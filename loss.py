import torch
import torch.nn as nn
import torch.nn.functional as F
import lightnet as ln

class CUDARegionLoss(nn.Module):
    def __init__(self, num_classes, anchors, network_stride):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = anchors.num_anchors
        # Convert Anchors object to tensor
        self.anchors = torch.tensor(anchors.anchors, dtype=torch.float32)
        self.network_stride = network_stride
        
        # Loss parameters
        self.coord_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.class_scale = 1.0
        self.thresh = 0.6
        
    def forward(self, output, target):
        """
        Forward pass of the loss function.
        
        Args:
            output: Network output tensor of shape (B, A*(5+C), H, W)
            target: Target tensor of shape (B, N, 5) where N is the number of objects
                   and each object is [class_id, cx, cy, w, h]
        """
        device = output.device
        
        # Ensure anchors are on the correct device
        self.anchors = self.anchors.to(device)
        
        # Get dimensions
        B, _, H, W = output.shape
        A = self.num_anchors
        C = self.num_classes
        
        # Reshape output
        output = output.view(B, A, 5 + C, H, W)
        output = output.permute(0, 1, 3, 4, 2).contiguous()
        
        # Split output into components
        coord = output[..., :4]  # (B, A, H, W, 4)
        conf = output[..., 4]    # (B, A, H, W)
        cls = output[..., 5:]    # (B, A, H, W, C)
        
        # Convert coordinates to absolute values
        grid_x = torch.arange(W, device=device).repeat(H, 1).view(1, 1, H, W)
        grid_y = torch.arange(H, device=device).repeat(W, 1).t().view(1, 1, H, W)
        
        pred_boxes = torch.zeros_like(coord, device=device)
        pred_boxes[..., 0] = (coord[..., 0].sigmoid() + grid_x) * self.network_stride
        pred_boxes[..., 1] = (coord[..., 1].sigmoid() + grid_y) * self.network_stride
        pred_boxes[..., 2] = coord[..., 2].exp() * self.anchors[:, 0].view(1, A, 1, 1)
        pred_boxes[..., 3] = coord[..., 3].exp() * self.anchors[:, 1].view(1, A, 1, 1)
        
        # Initialize target tensors
        coord_mask = torch.zeros(B, A, H, W, device=device)
        conf_mask = torch.ones(B, A, H, W, device=device) * self.noobject_scale
        cls_mask = torch.zeros(B, A, H, W, dtype=torch.bool, device=device)
        tcoord = torch.zeros(B, A, H, W, 4, device=device)  # Changed shape to match coord
        tconf = torch.zeros(B, A, H, W, device=device)
        tcls = torch.zeros(B, A, H, W, device=device)
        
        # Process each image in the batch
        for b in range(B):
            # Get ground truth boxes for this image
            gt = target[b]
            if gt.shape[0] == 0:
                continue
                
            # Convert ground truth to absolute coordinates
            gt_boxes = gt[:, 1:].to(device)  # [cx, cy, w, h]
            gt_classes = gt[:, 0].to(device)  # class_id
            
            # For each anchor
            for a in range(A):
                # Get predictions for this anchor
                cur_pred_boxes = pred_boxes[b, a].view(-1, 4)  # (H*W, 4)
                
                # Compute IoU between ground truth and predictions
                iou_gt_pred = self.compute_iou(gt_boxes, cur_pred_boxes)
                
                # Find best matching ground truth for each prediction
                best_iou, best_gt_idx = iou_gt_pred.max(dim=0)
                
                # Reshape best_iou to match the grid shape
                best_iou = best_iou.view(H, W)
                
                # Set confidence mask
                conf_mask[b, a] = self.noobject_scale
                conf_mask[b, a][best_iou > self.thresh] = self.object_scale
                
                # Set coordinate mask and target coordinates
                coord_mask[b, a][best_iou > self.thresh] = 1
                
                # Get indices where IoU > threshold
                mask_indices = best_iou > self.thresh
                
                # Set target coordinates only where mask is True
                tcoord[b, a, mask_indices] = gt_boxes[best_gt_idx.view(H, W)[mask_indices]]
                
                # Set class mask and target class
                cls_mask[b, a][mask_indices] = 1
                tcls[b, a][mask_indices] = gt_classes[best_gt_idx.view(H, W)[mask_indices]]
                
                # Set target confidence
                tconf[b, a][mask_indices] = 1
        
        # Compute losses
        # Coordinate loss
        coord_loss = F.mse_loss(coord * coord_mask.unsqueeze(-1), 
                              tcoord * coord_mask.unsqueeze(-1), 
                              reduction='sum')
        coord_loss = self.coord_scale * coord_loss
        
        # Confidence loss
        conf_loss = F.mse_loss(conf * conf_mask, 
                             tconf * conf_mask, 
                             reduction='sum')
        
        # Class loss
        if C > 1:
            cls_loss = F.cross_entropy(cls[cls_mask], 
                                     tcls[cls_mask].long(), 
                                     reduction='sum')
            cls_loss = self.class_scale * cls_loss
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = coord_loss + conf_loss + cls_loss
        
        return total_loss
    
    def compute_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes in center-width-height format.
        Both boxes should be on the same device (preferably CUDA).
        
        Args:
            boxes1: First set of boxes (N, 4) in [cx, cy, w, h] format
            boxes2: Second set of boxes (M, 4) in [cx, cy, w, h] format
        """
        # Convert to x1,y1,x2,y2 format
        boxes1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
        boxes1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
        boxes1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
        boxes1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2
        
        boxes2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
        boxes2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
        boxes2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
        boxes2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2
        
        # Compute intersection
        x1 = torch.max(boxes1_x1.unsqueeze(1), boxes2_x1.unsqueeze(0))
        y1 = torch.max(boxes1_y1.unsqueeze(1), boxes2_y1.unsqueeze(0))
        x2 = torch.min(boxes1_x2.unsqueeze(1), boxes2_x2.unsqueeze(0))
        y2 = torch.min(boxes1_y2.unsqueeze(1), boxes2_y2.unsqueeze(0))
        
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        # Compute union
        boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
        boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
        union = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - intersection + 1e-6
        
        return intersection / union
