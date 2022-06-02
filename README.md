# Object-Depth-detection-based-hybrid-Distance-estimator
We use our ODD model. Our purpose is that predict the distance between car based on Deep-Learning.  
-- It's not completed yet, and we're going to add speed calculation and voice later. --  


# Introduction
![ezgif com-gif-maker](https://user-images.githubusercontent.com/98331298/171547569-da221132-a13e-4b5f-8437-59cad290d3b2.gif)  
ODD 설명하는 글 (만들게 된 계기, 기대하는 바 등등)  
  
# Model Architecture
(모델 구조 사진)    
- 모델 설명  

# Performance
- Train data (# number of Data: 21,616)   

| Model | MAE | RMSE | Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 0.5057 | 0.8420 | 0.9599 |
| `XGBoost` | 0.2334 | 0.3149 | 0.9867 |  
| `LSTM` | 0.9027 | 1.7235 | 0.9807 |  
  
- Test data (# number of Data: 2,703)

| Model | MAE | RMSE | Accuracy | Pre-trained |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `RandomForest` | 1.3134 | 2.2927 | 0.9599 | |
| `XGBoost` | 1.2194 | 2.1536 | 0.9522 | |
| `LSTM` | 1.2200 | 2.2416 | 0.9522 | |

# Dataset
- 데이터셋 다운 받고 압축 푸는 경로 알려주기(드라이브 공유)  
(데이터 전처리하는 그 과정 사진)  
­- 데이터 만드는 법 적기(and 왜 그렇게 만들었는지 적기)  
  
# Training 
- 학습하는 방법 적기  

# Testing
- Opencv 활용해서 실행하는 법 적기  

 
