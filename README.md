# Object-Depth-detection-based-hybrid-Distance-estimator
We use our ODD model. Our purpose is that predict the distance between car based on Deep-Learning.  
-- It's not completed yet, and we're going to add speed calculation and voice later. --  


# Introduction
![ezgif com-gif-maker](https://user-images.githubusercontent.com/98331298/171547569-da221132-a13e-4b5f-8437-59cad290d3b2.gif)  
>Bounding box and Depth are extracted from image data to predict the distance.    
    
ODD 설명하는 글 (만들게 된 계기, 기대하는 바 등등)  
  
# Model Architecture
<img width="620" alt="architecture" src="https://user-images.githubusercontent.com/98331298/171553688-aee2e42a-9699-485a-8257-24f32b100ebe.png">
- 어떤 모델이 사용되었고, 무엇이 있는지 적기   

# Model Process
![ODD_process](https://user-images.githubusercontent.com/98331298/171548443-b4441f3e-7ac0-4108-913d-bd4e9db84fe3.jpg)  
- 데이터가 들어오면 어떻게 되는지 적기  

# Performance
**We use [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets. But we did not use the original data, We reconstructed the data to suit our purpose.**

------------
- **Train data (# number of Data: 21,616)**  

| Model | MAE | RMSE | Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 0.5057 | 0.8420 | 0.9807 |
| `XGBoost` | 0.2334 | 0.3149 | 0.9867 |  
| `LSTM` | 0.9027 | 1.7235 | 0.9657 |  
  
- **Test data (# number of Data: 2,703)**  

| Model | MAE | RMSE | Accuracy | Pre-trained |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 1.3134 | 2.2927 | 0.9492 | |
| `XGBoost` | 1.2194 | 2.1536 | 0.9522 | [XGBoost(Best)](https://drive.google.com/file/d/1YPiHMNylDWM2s_Q1_20BEnDYUcNgSu8H/view?usp=sharing) |
| `LSTM` | 1.2200 | 2.2416 | 0.9522 | [LSTM_612](https://drive.google.com/file/d/1q_u3PL0Ms99f5DI_YEGpB55OIItIMJQ-/view?usp=sharing) |
------------

# Dataset
1) Download Dataset
You can download the [KITTI Data(11.5GB)](https://drive.google.com/file/d/1MhDts48HWxIWPC7ZXLOMPqU2Mnt3NVmI/view?usp=sharing).  
Then, You unzip the data, and set the path.   
```
os.makedirs('./data/', exist_ok=True)  
'./datasets/data'  
```

2) Unzip Dataset
In the unzip folder, there is 'image' folder. So you move the folder into the 'data' folder.   
```
data
├── image                    
│   ├── test
|       ├── 000000.png            
│       ├── 000001.png
│       └── ...
│   ├── train             
│       ├── 000000.png             
│       ├── 000001.png            
│       └── ...                 
```
   
3) Make Our Datasets
We reconstructed data, because our final model, ZLE, use depth value of GLP-depth and bounding box of DETR. Apply the code below in order.  
```
1) kitti_detr_dataset_iou.py
2) kitti_glpdepth_dataset_iou.py
```
 
4) Split data
```
'./datasets/train_test_split.py'
```
  
# Training 
Look at the 'odd' folder, there are so many method, for example, LSTM, RandomForest, XGBoost.

# Testing
Use the file.
```
LSTM: ODD_application_LSTM.py
XGBoost: ODD_application.py
```

# References
[DETR](https://github.com/facebookresearch/detr)   
[GLP-depth](https://github.com/vinvino02/GLPDepth)   
[huggingface-transformers](https://github.com/huggingface/transformers)   
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/)  

 
