# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 06:08:23 2022

@author: Admin
"""

import pandas as pd
import numpy as np
from scipy import stats

# ODD process
class ODD:
    def __init__(self):
        pass
    
    # Make dataset
    def make_dataset_from_pretrained_model(self, scores, boxes, depth_map, DETR):
        self.data = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9])
        
        # BBOX input
        for p, (xmin, ymin, xmax, ymax) in zip(scores, boxes.tolist()):
            '''
            xmin, xmax 해서 본인 차선 range 안에 있는 object만 거리판단하기.
            '''
            #prt = True
            
            # class extraction
            cl = p.argmax()
            
            # class 설정
            classes = DETR.CLASSES[cl]
            if classes == 'motorcycle':
                classes = 'bicycle'
                
            elif classes == 'bus':
                classes = 'train'
                
            elif classes not in ['person', 'truck', 'car', 'bicycle', 'train']:
                classes = 'Misc'
                
            # color 맞추기
            if classes in ['Misc','person', 'truck', 'car', 'bicycle', 'train']:
                cl = ['Misc','person', 'truck', 'car', 'bicycle', 'train'].index(classes)
            else:
                continue
                
            # Detection rgb
            r,g,b = DETR.COLORS[cl][0] * 255, DETR.COLORS[cl][1] * 255, DETR.COLORS[cl][2] * 255
            rgb = (r,g,b)
            
            # Predict value1
            #x1 = xmin
            #y1 = ymin
            #x2 = xmax
            #y2 = ymax
            height = ymax - ymin
            width = xmax - xmin

            if int(xmin) < 0:
                xmin = 0
            if int(ymin) < 0:
                ymin = 0
                
            # Predict value2
            depth_mean = depth_map[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
            depth_median = np.median(depth_map[int(ymin):int(ymax),int(xmin):int(xmax)])
            depth_mean_trim = stats.trim_mean(depth_map[int(ymin):int(ymax), int(xmin):int(xmax)].flatten(), 0.2)
            depth_max = depth_map[int(ymin):int(ymax),int(xmin):int(xmax)].max() # ??
            #depth_min = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min() # ??
            #xy = np.where(prediction==depth_min) # ??
            #depth_x = xy[1][0]
            #depth_y = xy[0][0]
            
            data_list = pd.DataFrame(data=[xmin, ymin, xmax, ymax, depth_mean, depth_median, depth_max, depth_mean_trim, width, height, classes, rgb]).T
            self.data = pd.concat([self.data, data_list], axis=0)
            
            # 데이터 전처리
            self.data_preprocessing(depth_map)
            
            return self.data
            
            
    # 데이터 전처리 함수
    def data_preprocessing(self, prediction):
        self.data.index = [i for i in range(len(self.data))]
        
        xmin_list = [] ; ymin_list = [] ; xmax_list = [] ; ymax_list = []
        for k, (xmin, ymin, xmax, ymax) in zip(self.data.index, self.data[[0,1,2,3]].values):
            xmin_list.insert(0,xmin) ; ymin_list.insert(0,ymin) ; 
            xmax_list.insert(0,xmax) ; ymax_list.insert(0,ymax) ;
            #print(ymin_list)
                           
            for i in range(len(xmin_list)-1):
                y_range1 = np.arange(int(ymin_list[0]), int(ymax_list[0]+1)) # input image
                y_range2 = np.arange(int(ymin_list[i+1]), int(ymax_list[i+1]+1)) # 다른 image와 비교
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
                        
                        if area_intersect/area1 >= 0.70 or area_intersect/area2 >= 0.70: # 70% 이상 면적을 공유한다면
                            # 멀리 있는거 제거
                            if area1 < area2:
                                try:
                                    self.data.drop(index=k, inplace=True)
                                # 앞에서 미리 제거됬지만, list(xmin, ymin 등등)에 남아있는 경우
                                except:
                                    pass
                                
                            else:
                                try:
                                    self.data.drop(index=k-(i+1), inplace=True)
                                # 앞에서 미리 제거됬지만, list(xmin, ymin 등등)에 남아있는 경우
                                except:
                                    pass
                                
                        # 조금 겹친다면 depth_min and depth_mean 값 수정
                        elif  area_intersect/area1 > 0 or area_intersect/area2 > 0:
                            if area1 < area2:
                                prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                bbox = prediction[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                depth_mean = np.nanmean(bbox)
                                
                                if k in self.data.index:
                                    self.data.loc[k, 4] = depth_mean
                                
                            else:
                                prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                bbox = prediction[int(ymin_list[i+1]):int(ymax_list[i+1]), int(xmin_list[i+1]):int(xmax_list[i+1])]
                                depth_mean = np.nanmean(bbox)
                                
                                if k-(i+1) in self.data.index: 
                                    self.data.loc[k-(i+1), 4] = depth_mean
                                    
            # 인덱스 초기화
            self.data.reset_index(inplace=True)
            self.data.drop('index',inplace=True, axis=1)
            
            return self.data