# Improving Object Detection for YoloV2 with RGB+D fusion


## To-Do Steps

1. Loading models from lightnet framework (`model.py`) ![](https://geps.dev/progress/50) 
    1. Understand how model building works
    2. Code a function to call YoloV2 or YoloFusion from lightnet

2. Setup Loss Function (`loss.py`) ![](https://geps.dev/progress/50) 
    1. Understand how lightnet RegionLoss works
    2. Code a function to call RegionLoss from lightnet

3. Dataset Object (`dataset.py`) ![](https://geps.dev/progress/0) 
    1. Understand KITTI annotations
    2. Understand the YoloV2 expected annotation format
    3. Code a function to convert KITTI for YoloV2 lightnet annotation format
    4. Code the `DatasetObject()` class

4. Train Loop (`train.py`) ![](https://geps.dev/progress/0) 
    1. Understanding how YoloV2 fine tuning workl
    2. Code `train_step()` function
    3. Code `train` function 

5. Metrics (`utils.py`) ![](https://geps.dev/progress/0) 
    1. Code metrics
    2. ...
