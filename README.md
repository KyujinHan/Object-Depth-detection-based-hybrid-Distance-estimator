# [Vehicle Distance Estimation from a Monocular Camera for Advanced Driver Assistance Systems]()
We use our VDE(ODD) model. Our purpose is that predict the distance between car based on Deep-Learning.  
>(before name)
>Object-Depth-detection-based-hybrid-Distance-estimator (Called, ODD)  
>We will more update the github readme.  
>(11/04) Paper is not public yet. Wait a month(?)  


# Introduction
![ezgif com-gif-maker](https://user-images.githubusercontent.com/98331298/171547569-da221132-a13e-4b5f-8437-59cad290d3b2.gif)  
>Bounding box and Depth are extracted from image data to predict the distance.    
>
    
  
# Model Process
![odd_framework](https://user-images.githubusercontent.com/98331298/199839808-ec393b0d-ffce-4dc3-8ce1-d9c2f07b64c6.png)
> Now, we also called 'VDE'.  
  
# Performance
**We use [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets. But we did not use the original data, We reconstructed the data to suit our purpose.**  

------------
- **Train data (# number of Data: 21,616)**  

| Model | MAE | RMSE | Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 0.5057 | 0.8420 | 0.9807 |
| `XGBoost` | 0.2334 | 0.3149 | 0.9867 |  
| `LSTM` | 0.6988 | 1.4736 | 0.9746 |  
  
- **Test data (# number of Data: 2,703)**  

| Model | MAE | RMSE | Accuracy | Pre-trained | scaler file |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 1.3134 | 2.2927 | 0.9492 | | |
| `XGBoost` | 1.2194 | 2.1536 | 0.9522 | [XGBoost](https://drive.google.com/file/d/1YPiHMNylDWM2s_Q1_20BEnDYUcNgSu8H/view?usp=sharing) | (will update) |
| `LSTM` | 1.1658 | 2.1420 | 0.9526 | [LSTM_612 // not update](https://drive.google.com/file/d/1DqtP08KgLiUrbPnrXuSZFC7Tr55IAHS_/view?usp=sharing) | (will update) |  

------------

**More detail performace, you can find our [paper]().**

# Dataset
1) Download Dataset
You can download the [KITTI Data](https://drive.google.com/file/d/1MhDts48HWxIWPC7ZXLOMPqU2Mnt3NVmI/view?usp=sharing).  
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
# You must check the saving file name.
```
 
4) Split data
```
'./datasets/train_test_split.py'
# You must check the saving file name.
```
  
# Training 
Look at the 'odd' folder, there are so many method, for example, LSTM, RandomForest, XGBoost.

# Testing
Use the file.
```
# Before, implementing file, you need some file below.
# :> Model weight(.pth file), scaler file
# You can download in 'Performace block'.

# And, implementing this file.
LSTM: ODD_application_LSTM.py
XGBoost: ODD_application.py
```

# References
[DETR](https://github.com/facebookresearch/detr)   
[GLP-depth](https://github.com/vinvino02/GLPDepth)   
[huggingface-transformers](https://github.com/huggingface/transformers)   
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/)  

 
# Citation
```
Thank you so much for your interest in our model.
```

```
we will make it.
```
