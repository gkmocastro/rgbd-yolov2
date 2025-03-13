import lightnet as ln
import functools
import torch
from torch import nn
from utils import rename_state_dict
from remapping_rules import remap_rgbd_15_fusion_model, remap_rgbd_15_fusion_layers

def model_builder(num_classes, model_type="rgb", fuse_layer=16):
    """ 
    Function to return the specified Yolo Model from lightnet framework

    - under development... 
    """
    
    if model_type=="rgb":
        detection_model = ln.models.YoloV2(num_classes)
        detection_model.load('weights/yolo-pretrained_darknet.pt', strict=False)
        return detection_model
    elif model_type=="depth":
        detection_model = ln.models.YoloV2(num_classes)
        detection_model.load('weights/yolo-pretrained_darknet.pt', strict=False)
        #original_conv = detection_model.backbone.module[0]
        new_conv = ln.network.layer.Conv2dBatchAct(in_channels=1, 
                                           out_channels=32, 
                                           kernel_size=(3, 3), 
                                           stride=(1, 1), 
                                           padding=(1, 1), 
                                           activation=functools.partial(nn.modules.activation.LeakyReLU, negative_slope=0.1, inplace=True))
        print(f"Replacing first layer with new conv: {new_conv}")
        detection_model.backbone.module[0] = new_conv
        return detection_model
    elif model_type=="rgbd":
        fusion_model = ln.models.YoloFusion(num_classes=3, fuse_layer=fuse_layer)
        rgb_state_dict = torch.load("models/rgb_state_dict.pth", weights_only=True)
        renamed_rgb_state_dict = rename_state_dict(rgb_state_dict, remap_rgbd_15_fusion_model)
        fusion_model.load_state_dict(renamed_rgb_state_dict, strict=False)

        depth_state_dict = torch.load("models/depth_99.pth", weights_only=True)
        renamed_depth_state_dict = rename_state_dict(depth_state_dict, remap_rgbd_15_fusion_layers)
        fusion_model.load_state_dict(renamed_depth_state_dict, strict=False)

        ## test equality of a single layer, must change to test all
        fusion_weight = fusion_model.state_dict()['layers.1.regular.11_convbatch.layers.1.weight'].cpu()
        rgb_weight = rgb_state_dict['backbone.module.11_convbatch.layers.1.weight'].cpu()
        if torch.equal(fusion_weight, rgb_weight):
            print("Weights equivalent")
        else: 
            print("Weights not equivalent")

    
        return fusion_model
    else: 
        print("Model type is not valid. Try rgb, depth or rgbd.")
        return None