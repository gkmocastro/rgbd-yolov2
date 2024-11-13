# Improving Object Detection for YoloV2 with RGB+D fusion

## Base Paper

This repository aims to replicate the research from the paper **Exploring RGB+Depth Fusion for Real-Time
Object Detection** by Tanguy Ophoff, Kristof Van Beeck and Toon Goedemé. 

    @article{ophoff2019exploring,
    title={Exploring RGB+ Depth fusion for real-time object detection},
    author={Ophoff, Tanguy and Van Beeck, Kristof and Goedem{\'e}, Toon},
    journal={Sensors},
    volume={19},
    number={4},
    pages={866},
    year={2019},
    publisher={MDPI}
    }

Adding to the research, our research aims to test depth maps estimated using the SOTA monocular depth estimators (e.g. Midas, Marigold, Depth Anything) to check further improvements. 

## To-Do Steps


### Part 1: Code the project

1. Loading models from lightnet framework (`model.py`) ![](https://geps.dev/progress/80) 
    1. [x] Understand how model building works
    2. [ ] Code a function to call YoloV2 or YoloFusion from lightnet
    3. [ ] Build a remapping function to load the darknet19 pre-trained weights into the correct layers of YoloFusion

2. Setup Loss Function (`loss.py`) ![](https://geps.dev/progress/100) 
    1. [x] Understand how lightnet RegionLoss works
    2. [x] Code a function to call RegionLoss from lightnet

3. Dataset Object (`dataset.py`) ![](https://geps.dev/progress/60) 
    1. [x] Understand KITTI annotations 
    2. [x] Understand the YoloV2 expected annotation format
    3. [x] Code a function to convert KITTI for YoloV2 lightnet annotation format
    4. [ ] Code the `DatasetObject()` class
    5. [ ] Adapt the `DatasetObject()` to work with RGBD data
    6. [ ] Augmentation

4. Train Loop (`train.py`) ![](https://geps.dev/progress/50) 
    1. [ ] Understanding how YoloV2 fine tuning works
    2. [x] Code `train_step()` function
    3. [x] Code `train` function 
    4. [ ] Layer Freezing

5. Metrics (`utils.py`) ![](https://geps.dev/progress/30) 
    1. [ ] Code metrics
    2. [x] Results visualization  
   


### Part 2: Training and testing models

1. Train and test the RGB-only KITTI YoloV2 Model ![](https://geps.dev/progress/30) 
2. Train and test the Depth-only KITTI YoloV2 Model
   1. Run the depth estimator on the dataset
   2. Modify the network (first layer only, as the paper says)
3. Train and test a single RGBD YoloFusion KITTI Model
4. Train and test many RGBD YoloFusion KITTI Model changing the fusion layer
   
### Part 3: Writing

1. Introduction
2. Related works
3. Methods
4. Results and discussion
5. Conclusion
   

