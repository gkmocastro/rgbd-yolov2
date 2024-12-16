import lightnet as ln
import functools
from torch import nn

def model_builder(num_classes, model_type="rgb", fuse_layer=16):
    """ 
    Function to return the specified Yolo Model from lightnet framework

    - under development... 
    """
    
    if model_type=="rgb":
        detection_model = ln.models.YoloV2(num_classes)
        detection_model.load('weights/yolo-pretrained_darknet.pt', strict=False)

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

    elif model_type=="rgbd":
        detection_model = ln.models.YoloFusion(num_classes=3, fuse_layer=fuse_layer)

    
    return detection_model