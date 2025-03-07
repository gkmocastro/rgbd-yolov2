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

1. Loading models from lightnet framework (`model.py`) ![](https://geps.dev/progress/100) 
    1. [x] Understand how model building works
    2. [X] Code a function to call YoloV2 or YoloFusion from lightnet
    3. [X] Build a remapping function to load the darknet19 pre-trained weights into the correct layers of YoloFusion

2. Setup Loss Function (`loss.py`) ![](https://geps.dev/progress/100) 
    1. [x] Understand how lightnet RegionLoss works
    2. [x] Code a function to call RegionLoss from lightnet

3. Dataset Object (`dataset.py`) ![](https://geps.dev/progress/100) 
    1. [x] Understand KITTI annotations 
    2. [x] Understand the YoloV2 expected annotation format
    3. [x] Code a function to convert KITTI for YoloV2 lightnet annotation format
    4. [x] Code the `DatasetObject()` class
    5. [X] Adapt the `DatasetObject()` to work with RGBD data

4. Train Loop (`train.py`) ![](https://geps.dev/progress/100) 
    1. [X] Understanding how YoloV2 fine tuning works
    2. [x] Code `train_step()` function
    3. [x] Code `train` function 

5. Metrics (`utils.py`) ![](https://geps.dev/progress/100) 
    1. [X] Code metrics
       1. [x] Implement a mAP function
       2. [X] calculate mAP in the test_step
    2. [x] Results visualization  
    3. [X] Implement the metric as the base paper
   


### Part 2: Training and testing models

1. Train and test the RGB-only KITTI YoloV2 Model ![](https://geps.dev/progress/50) 
2. Train and test the Depth-only KITTI YoloV2 Model ![](https://geps.dev/progress/50) 
   1. Using Marigold Dataset (Train :white_check_mark: | Test)
   2. Using Depth Anything v2 Dataset (Train :white_check_mark: | Test)
   3. Using Midas Dataset (Train | Test)
3. Train and test a single RGBD YoloFusion KITTI Model ![](https://geps.dev/progress/50) 
   1. Using Marigold Dataset (Train :white_check_mark: | Test)
   2. Using Depth Anything v2 Dataset (Train :white_check_mark: | Test)
   3. Using Midas Dataset 
4. Train and test many RGBD YoloFusion KITTI Model changing the fusion layer (optional) ![](https://geps.dev/progress/0)
   
### Part 3: Writing

1. Introduction
2. Related works
3. Methods
4. Results and discussion
5. Conclusion
   

