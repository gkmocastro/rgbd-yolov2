import lightnet as ln
import functools
import torch
from torch import nn
from remapping_rules import remap_rgb_branch_15, remap_depth_branch_15


def rename_state_dict(state_dict, layer_mapping):
    renamed_state_dict = {}
    for src_layer, tgt_layer in layer_mapping.items():
        if src_layer in state_dict:
            renamed_state_dict[tgt_layer] = state_dict[src_layer]
        else:
            print(f"Warning: {src_layer} not found in the source state_dict.")
    return renamed_state_dict

def check_weights(rgbd_model, rgb_model, depth_model, rgb_remap, depth_remap):
    """
    Function to check if the weights of the RGBD model are equivalent to the sum of the RGB and Depth models.
    """
    for rgb_param_name, rgbd_param_name in rgb_remap.items():
        rgb_param = rgb_model[rgb_param_name]
        rgbd_param = rgbd_model.state_dict()[rgbd_param_name]
        if torch.equal(rgb_param.cpu(), rgbd_param.cpu()):
            #print(f"RGB Weights are equivalent. \n{rgb_param_name}\n{rgbd_param_name}\n")
            last_check = True
        else:
            print(f"Weights are NOT equivalent.\n{rgb_param_name}\n{rgbd_param_name}\n")
            last_check = False
            
    if last_check:
        print("All RGB weights are equivalent.\n")        

    for depth_param_name, rgbd_param_name in depth_remap.items():
        depth_param = depth_model[depth_param_name]
        rgbd_param = rgbd_model.state_dict()[rgbd_param_name]
        if torch.equal(depth_param.cpu(), rgbd_param.cpu()):
            #print(f"Depth Weights are equivalent. \n{depth_param_name}\n{rgbd_param_name}\n")
            last_check = True
        else:
            #print(f"Depth Weights are NOT equivalent.\n{depth_param_name}\n{rgbd_param_name}\n")
            last_check = False
            

    if last_check:
        print("All Depth weights are equivalent.") 



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
        renamed_rgb_state_dict = rename_state_dict(rgb_state_dict, remap_rgb_branch_15)
        fusion_model.load_state_dict(renamed_rgb_state_dict, strict=False)

        depth_state_dict = torch.load("models/depth_99.pth", weights_only=True)
        renamed_depth_state_dict = rename_state_dict(depth_state_dict, remap_depth_branch_15)
        fusion_model.load_state_dict(renamed_depth_state_dict, strict=False)

        ## test equality of a single layer, must change to test all
        check_weights(fusion_model, rgb_state_dict, depth_state_dict, remap_rgb_branch_15, remap_depth_branch_15)
        # fusion_weight = fusion_model.state_dict()['layers.1.regular.11_convbatch.layers.1.weight'].cpu()
        # rgb_weight = rgb_state_dict['backbone.module.11_convbatch.layers.1.weight'].cpu()
        # if torch.equal(fusion_weight, rgb_weight):
        #     print("Weights equivalent")
        # else: 
        #     print("Weights not equivalent")

    
        return fusion_model
    else: 
        print("Model type is not valid. Try rgb, depth or rgbd.")
        return None
    