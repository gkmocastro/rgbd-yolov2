import lightnet as ln


def model_builder(num_classes):
    """ 
    Function to return the specified Yolo Model from lightnet framework

    - under development... 
    """
    detection_model = ln.models.YoloV2(num_classes)
    detection_model.load('weights/yolo-pretrained_darknet.pt', strict=False)
    return detection_model