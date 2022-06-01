# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:55:25 2022

@author: ODD Team
# GLPdepth을 이용해서 DETR에서 뽑은 BBOX의 DEPTH info 뽑아서 저장하기
"""

# 1. Import Module
import os
import pandas as pd
import numpy as np
import numpy
import torch
import cv2
import time
from scipy import stats
from tqdm import tqdm
from PIL import Image
from model.glpdepth import GLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################################
# 2. Data 및 변수 세팅
glp_kitti_preprocessing_data = pd.read_csv('./datasets/detr_kitti_preprocessing_data_iou_remove2.csv')
train_image_list = os.listdir('./datasets/data/image/train') # 7481

################################################################################################################################
# 3. Model 불러오기
# GLPdepth 불러오기
glp_pretrained = 'vinvino02/glpn-kitti'
GLPdepth = GLP(glp_pretrained)
GLPdepth.model.eval()
GLPdepth.model.to(device)

################################################################################################################################
# 4. Algorithm (Make data)
# 내가 원하는 이미지
'''
detr에서 얻은 KITTI와 동일한 객체 BBOX를 기준으로 그 안의 DEPTH 정보(min, mean)을 뽑아낸다.

min: GLPdepth은 거리가 가까울 수록 수치가 낮기 때문에 min을 이용해본다.
max: DPT은 거리가 가까울 수록 수치가 크기 때문에 max를 이용해본다.
'''

start = time.time() # 시간 측정 시작

# 데이터 프레임 저장할 곳
depth_mean = []
depth_min = []
depth_median = []
depth_max = []
depth_coordinate = []
depth_x = []
depth_y = []
depth_info = pd.DataFrame(columns={'depth_min','depth_mean','depth_x','depth_y'})

for k in tqdm(range(len(train_image_list))): # 7481개의 데이터
    # 진행 상황 알라기
    print('이미지 전체 {} 중 {}번째 진행중'.format(len(train_image_list), k+1))
    
    # k번째 이미지
    filename = train_image_list[k]
    
    img = Image.open(os.path.join('./datasets/data/image/train/',filename))
    img_shape = cv2.imread(os.path.join('./datasets/data/image/train/',filename)).shape
    
    df_choose = glp_kitti_preprocessing_data[glp_kitti_preprocessing_data['filename']==filename]
    coordinates_array = df_choose[['xmin','ymin','xmax','ymax']].values
    
    # Make depth map
    prediction = GLPdepth.predict(img, img_shape)
    # append list

    for (xmin, ymin, xmax, ymax) in coordinates_array:
        
        # depth map의 index는 최소한 0
        if int(xmin) < 0:
            xmin = 0
        if int(ymin) < 0:
            ymin = 0
             
        depth_mean_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
        depth_min_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min()
        depth_median_info = np.median(prediction[int(ymin):int(ymax),int(xmin):int(xmax)])
        depth_max_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].max()
        xy = np.where(prediction==depth_min_info)
        
        '''
        glp_kitti_data_1.csv 의 경우 추가 
        '''
        y_coor = (ymax-ymin)*0.2
        #depth_coordinate_info = prediction[int(ymin+y_coor):int((ymin+ymax)/2 - y_coor), int((xmin+xmax)*0.5)].min()
        # 15% 절사평균
        depth_mean_trim = stats.trim_mean(prediction[int(ymin):int(ymax), int(xmin):int(xmax)].flatten(), 0.2)
        
        depth_x.append(xy[1][0])
        depth_y.append(xy[0][0])
        depth_mean.append(depth_mean_info)
        depth_min.append(depth_min_info)
        depth_median.append(depth_median_info)
        depth_max.append(depth_max_info)
        depth_coordinate.append(depth_mean_trim)
        
# 인덱스 재설정
glp_kitti_preprocessing_data.reset_index(inplace=True)
glp_kitti_preprocessing_data.drop('index', axis=1, inplace=True)

# 데이터 저장
depth_info['depth_mean'] = depth_mean
depth_info['depth_min'] = depth_min
depth_info['depth_x'] = depth_x
depth_info['depth_y'] = depth_y
depth_info['depth_median'] = depth_median
depth_info['depth_max'] = depth_max
depth_info['depth_mean_trim'] = depth_coordinate 

# NA 값 확인
depth_info.isnull().sum(axis=0)

# 데이터 병합
glp_kitti_preprocessing_data = pd.concat([glp_kitti_preprocessing_data, depth_info], axis=1)
        

# IOU function with BBOX (real)
ind = []
for filename in tqdm(glp_kitti_preprocessing_data['filename'].unique()):
    #filename = '002961.png'
    # GLPdepth model
    img = Image.open(os.path.join('./datasets/data/image/train/',filename))
    img_shape = cv2.imread(os.path.join('./datasets/data/image/train/',filename)).shape
    prediction = GLPdepth.predict(img, img_shape)
    
    k = 1
    df_sample = glp_kitti_preprocessing_data[glp_kitti_preprocessing_data['filename']==filename]
    
    xmin_list = [] ; ymin_list = [] ; xmax_list = [] ; ymax_list = []
    
    if len(df_sample) == 1:
        continue
    
    else:
        for (xmin, ymin, xmax, ymax) in df_sample[['xmin','ymin','xmax','ymax']].values:
            xmin_list.insert(0,xmin) ; ymin_list.insert(0,ymin) ; 
            xmax_list.insert(0,xmax) ; ymax_list.insert(0,ymax) ;
            #print(ymin_list)
            
            if k == 1:
                k += 1
                continue
                
            elif k >= 2: 
                for i in range(len(xmin_list)-1):
                    y_range1 = np.arange(int(ymin_list[0]), int(ymax_list[0]+1))
                    y_range2 = np.arange(int(ymin_list[i+1]), int(ymax_list[i+1]+1))
                    y_intersect = np.intersect1d(y_range1, y_range2)
                    
                    #print(y_intersect)
                    
                    if len(y_intersect) >= 1: 
                        x_range1 = np.arange(int(xmin_list[0]), int(xmax_list[0])+1)
                        x_range2 = np.arange(int(xmin_list[i+1]), int(xmax_list[i+1]+1))
                        x_intersect = np.intersect1d(x_range1, x_range2)
                        
                        #print(x_intersect)
                        
                        if len(x_intersect) >= 1: # BBOX가 겹친다면 밑에 구문 실행
                            area1 = (y_range1.max() - y_range1.min())*(x_range1.max() - x_range1.min())
                            area2 = (y_range2.max() - y_range2.min())*(x_range2.max() - x_range2.min())
                            area_intersect = (y_intersect.max() - y_intersect.min())*(x_intersect.max() - x_intersect.min())
                            
                            zloc1 = df_sample[df_sample['ymin']==ymin_list[0]]['zloc'].values[0]
                            zloc2 = df_sample[df_sample['ymin']==ymin_list[i+1]]['zloc'].values[0]
                            
                            if area_intersect/area1 >= 0.70 or area_intersect/area2 >= 0.70: # 70% 이상 면적을 공유한다면
                                
                                # 가장 멀리 있는 인덱스 추출
                                if zloc1 > zloc2:
                                    ind.append(df_sample[df_sample['zloc']==zloc1].index[0])
                                else:
                                    ind.append(df_sample[df_sample['zloc']==zloc2].index[0])
                            
                            # 조금 겹친다면 depth_min and depth_mean 값 수정
                            elif  area_intersect/area1 > 0 or area_intersect/area2 > 0:
                                if zloc1 > zloc2:
                                    prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                    bbox = prediction[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                    re_depth_min = np.nanmin(bbox)
                                    re_depth_mean = np.nanmean(bbox)
                                    
                                    #print(bbox)
                                    #print(re_depth_min)
                                    
                                    index1 = df_sample[df_sample['zloc']==zloc1].index[0]
                                    #print(index1)
                                    glp_kitti_preprocessing_data.loc[index1, 'depth_min'] = re_depth_min
                                    glp_kitti_preprocessing_data.loc[index1, 'depth_mean'] = re_depth_mean
                                
                                    
                                else:
                                    prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                    bbox = prediction[int(ymin_list[i+1]):int(ymax_list[i+1]), int(xmin_list[i+1]):int(xmax_list[i+1])]
                                    re_depth_min = np.nanmin(bbox)
                                    re_depth_mean = np.nanmean(bbox)
                                    
                                    #print(re_depth_min)
                                    
                                    index2 = df_sample[df_sample['zloc']==zloc2].index[0]
                                    glp_kitti_preprocessing_data.loc[index2, 'depth_min'] = re_depth_min
                                    glp_kitti_preprocessing_data.loc[index2, 'depth_mean'] = re_depth_mean
                
    #break
                                    

# 인덱스 드랍
print(len(ind))
glp_kitti_preprocessing_data.drop(index=np.unique(ind), inplace=True)         
print(glp_kitti_preprocessing_data.isnull().sum(axis=0))
print(len(glp_kitti_preprocessing_data))
glp_kitti_preprocessing_data.dropna(subset=['depth_min'], axis=0, inplace=True)      
glp_kitti_preprocessing_data.dropna(subset=['depth_mean'], axis=0, inplace=True)            
print(glp_kitti_preprocessing_data.isnull().sum(axis=0))
                 
# 인덱스 재설정
glp_kitti_preprocessing_data.reset_index(inplace=True)
glp_kitti_preprocessing_data.drop('index', axis=1, inplace=True) 

# Outlier 제거
#mask1 = (glp_kitti_preprocessing_data['depth_mean']-0.0930*glp_kitti_preprocessing_data['zloc'])+1.116 > 0
#mask2 = (glp_kitti_preprocessing_data['depth_mean']-0.13*glp_kitti_preprocessing_data['zloc'])-2.5 < 0
#mask3 = (mask1 & mask2)           

#glp_kitti_preprocessing_data = glp_kitti_preprocessing_data[mask3]         

# Check time
print('Finish')
end = time.time() # 시간 측정 끝
print(f"{end - start:.5f} sec") # 2100 sec.


# 데이터 저장 (최종)
#glp_kitti_preprocessing_data.to_csv('./datasets/glp_kitti_data.csv', index=False)
glp_kitti_preprocessing_data.to_csv('./datasets/glp_kitti_data_iou.csv', index=False)

print(len(glp_kitti_preprocessing_data))