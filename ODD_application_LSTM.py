# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:55:32 2022

@author: ODD_team

It is used zloc model as LSTM, little change, but systemically there is no change with XGboost application model
"""

import os
import pandas as pd
import numpy as np
import pickle
import time
import torch
import cv2
from model.detr import DETR
from model.glpdepth import GLP
from model.lstm import LSTM
import warnings
from PIL import Image
from scipy import stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################# Start ###########################
warnings.filterwarnings(action='ignore')


'''
function define
'''
# This is not use function, yet.

# sound1
'''
import pyttsx3 
s= pyttsx3.init()
speak = "전방을 주의하세요"
s.say(speak)
s.runAndWait()
'''

# sound2
'''
import winsound as sd
def beepsound():
    fr = 2500
    du = 700
    sd.Beep(fr,du)
'''

# Calculate velocity
'''
count = 0
# 속력 측정
def speed_estimate(prev,current_v, time):
    diff = abs(prev-current_v) # 이동거리 m/0.3s
    # 1 : 1/3600 = 0.3 : x
    # 1/3600 * (10/3) = x
    # diff * 0.001 * (0.3/0.00093)
    
    #velocity = (diff*3.6)*time # 0.3sec 단위로
    velocity = (diff/time) * 3.6
    return float(velocity)


def odd_process(zloc, speed):
    if speed>=80: #여기서 스피드는 속력이 아니라 속도(상대적 속도임)
        if zloc<50:
            print('실행 80')
            s.say(speak)
            s.runAndWait()
            
    elif speed>=40:
        if zloc<30:
            print('실행 40')            
            s.say(speak)
            s.runAndWait()

    elif speed>=10:
        if zloc<10:     
            print('실행 10')
            s.say(speak)
            s.runAndWait()
    else:
        pass
'''               

                
'''
Model 및 카메라 정의
'''
prepare_start = time.time()
##############################################################################################################################################
# 모델 정의
# DETR 불러오기
model_path = 'facebookresearch/detr:main'
model_backbone = 'detr_resnet101'
#sys.modules.pop('models') # ModuleNotFoundError: No module named 'models.backbone' 이 에러 발생시 수행
DETR = DETR(model_path, model_backbone)
DETR.model.eval()
DETR.model.to(device)

# GLPdepth 불러오기
glp_pretrained = 'vinvino02/glpn-kitti'
GLPdepth = GLP(glp_pretrained)
GLPdepth.model.eval()
GLPdepth.model.to(device)

# Z-location Estimator 불러오기 (LSTM)
lstm_path = './weights/ODD_variable16.pth'
ZlocE = LSTM(lstm_path)
ZlocE.model.eval()
ZlocE.model.to(device)

'''
variable which we used
: [xmin, ymin, xmax, ymax, width, height, depth_mean_trim, depth_mean, depth_median, Misc, bicycle, car, person, train, truck]

'''
#스케일러 불러오기
scaler = pickle.load(open('./weights/lstm_scaler.pkl', 'rb'))
##############################################################################################################################################


# 카메라 정의
#cap = cv2.VideoCapture('./test_video/object_video2.mp4')
cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
os.makedirs('./test_video/output', exist_ok=True)
os.makedirs('./test_video/frame', exist_ok=True)
#out = cv2.VideoWriter('./test_video/output/ODD_test.mp4', fourcc, 30.0, (1242,374))
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1242) # 가로
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 374) # 세로

prepare_end = time.time()
print(prepare_end - prepare_start)
'''
# 비디오 작동하기
'''
currentframe = 1
if cap.isOpened():
    while(True):
        ret, frame= cap.read()
        if ret:
            start = time.time() # 시간 측정 시작
            #cv2.imshow("webcam",frame)
            
            # 테스트를 위해 임시로 넣음.
            name = './test_video/frame/object_video2_'+str(currentframe)+'.jpg'
            
            if cv2.waitKey(1) != -1:
                #cv2.imwrite('webcam_snap.jpg',frame)
                break
            #정상적인 케이스임
            #first_step = detr_model(frame)
            #second_step =GLPdepth(frame,first_step)
            #speed="계산 방법"
            #zloc= xgb_model.predict("여기서는 들어가는 최종 텐서를 맞추어서 넣어주면됨.")
            #odd_process(zloc,speed)
            
            cv2.imwrite(name, frame) # 이미지 save
            currentframe += 1
            
            '''
            Step1) Image DETR 적용
            '''
            frame = cv2.resize(frame, (1280, 640))
            color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            img_shape = color_coverted.shape[0:2]
            
            # Predicted
            time1 = time.time()
            scores, boxes = DETR.detect(pil_image) # Detection
            time2 = time.time()
            print('DETR')
            print(time2 - time1)
            
            '''
            Step2) GLP_Depth 적용
            '''
            # Make depth map
            time1 = time.time()
            prediction = GLPdepth.predict(pil_image, img_shape)
            time2 = time.time()
            print('GLP')
            print(time2 - time1)
            
            '''
            Step3) 입력 및 z-model 적용
            '''
            data = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9])
            # BBOX input
            for p, (xmin, ymin, xmax, ymax) in zip(scores, boxes.tolist()):
                '''
                xmin, xmax 해서 본인 차선 range 안에 있는 object만 거리판단하기.
                '''
                prt = True
                
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
                x1 = xmin
                y1 = ymin
                x2 = xmax
                y2 = ymax
                height = ymax - ymin
                width = xmax - xmin

                if int(xmin) < 0:
                    xmin = 0
                if int(ymin) < 0:
                    ymin = 0
                    
                # Predict value2
                depth_mean = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
                depth_median = np.median(prediction[int(ymin):int(ymax),int(xmin):int(xmax)])
                depth_mean_trim = stats.trim_mean(prediction[int(ymin):int(ymax), int(xmin):int(xmax)].flatten(), 0.2)
                depth_max = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].max() 
                #depth_min = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min() # ??
                #xy = np.where(prediction==depth_min) # ??
                #depth_x = xy[1][0]
                #depth_y = xy[0][0]
                
                data_list = pd.DataFrame(data=[xmin, ymin, xmax, ymax, width, height, depth_mean_trim, depth_mean, depth_max, depth_median, classes, rgb]).T
                data = pd.concat([data, data_list], axis=0)
                #print(data.info())
            
            '''
            전처리
            preprocessing
            bbox 비교해서 70% 이상 겹친다면 그 뒤에 있는 영역을 지우고,
            if our image are overlap over 70% we remove futher object 
            만약 아니라면, 겹친 부분을 제외한 후, 다시 depth를 계산해서 값 출력
            if not, exclude overlapped and calculate depth again.
            '''
            
            data.index = [i for i in range(len(data))]
            
            xmin_list = [] ; ymin_list = [] ; xmax_list = [] ; ymax_list = []
            for k, (xmin, ymin, xmax, ymax) in zip(data.index, data[[0,1,2,3]].values):
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
                                        data.drop(index=k, inplace=True)
                                    # 앞에서 미리 제거됬지만, list(xmin, ymin 등등)에 남아있는 경우
                                    except:
                                        pass
                                    
                                else:
                                    try:
                                        data.drop(index=k-(i+1), inplace=True)
                                    # 앞에서 미리 제거됬지만, list(xmin, ymin 등등)에 남아있는 경우
                                    except:
                                        pass
                                    
                            # 조금 겹친다면 depth_min and depth_mean 값 수정
                            elif  area_intersect/area1 > 0 or area_intersect/area2 > 0:
                                if area1 < area2:
                                    prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                    bbox = prediction[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                    depth_mean = np.nanmean(bbox)
                                    
                                    if k in data.index:
                                        data.loc[k, 4] = depth_mean
                                    
                                else:
                                    prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                    bbox = prediction[int(ymin_list[i+1]):int(ymax_list[i+1]), int(xmin_list[i+1]):int(xmax_list[i+1])]
                                    depth_mean = np.nanmean(bbox)
                                    
                                    if k-(i+1) in data.index: 
                                        data.loc[k-(i+1), 4] = depth_mean
                                    
                                    
                                   
            
            # 인덱스 초기화
            data.reset_index(inplace=True)
            data.drop('index',inplace=True, axis=1)
            
            # input text & draw bbox
            distance = []
            for k in data.index:
                x_range = np.arange(int(data.iloc[k,0]), int(data.iloc[k,2])+1) # xmax~xmin 
                line_range = np.arange(500, 742+1)
                
                # 겹칠 때 판단하기
                if len(np.intersect1d(x_range, line_range)) >= 10: 
                    classes = data.iloc[k,-2] # class info
                    '''
                    Z-model 적용
                    '''
                    #Misc, bicycle, car, person, train, truck
                    if classes == 'Misc':
                        array = torch.tensor([[1,0,0,0,0,0]], dtype=torch.float32)
                    elif classes == 'bicycle':
                        array = torch.tensor([[0,1,0,0,0,0]], dtype=torch.float32)
                    elif classes == 'car':
                        array = torch.tensor([[0,0,1,0,0,0]], dtype=torch.float32)
                    elif classes == 'person':
                        array = torch.tensor([[0,0,0,1,0,0]], dtype=torch.float32)
                    elif classes == 'train':
                        array = torch.tensor([[0,0,0,0,1,0]], dtype=torch.float32)
                    elif classes == 'truck':
                        array = torch.tensor([[0,0,0,0,0,1]], dtype=torch.float32)
                    #input_data = torch.tensor([[x1,y1,x2,y2,depth_mean,depth_median, depth_max, depth_mean_trim, width, height]])
                    #input_data_scaler = torch.tensor(scaler.transform(input_data)) # scaler 적용
                    
                    input_data = np.array(data.iloc[[k],0:10].values, dtype=np.float32)
                    input_data = torch.from_numpy(input_data)
                    input_data = torch.cat([input_data, array], axis=1)
                    model_data = torch.tensor(scaler.transform(input_data), dtype=torch.float32)
                    
                    #model_data = torch.cat([input_data_scaler, array], dim=1)
                    #dataframe = pd.DataFrame(model_data,columns=[0,1,2,3,4,5,6,7,8,9,'Misc','bicycle','car','person','train','truck'])
                    
                    # Predict
                    preds = ZlocE.predict(model_data).detach().numpy()[0]
                    
                    # error1: 좌표는 int형.
                    cv2.rectangle(frame, (int(data.iloc[k,0]), int(data.iloc[k,1])), (int(data.iloc[k,2]), int(data.iloc[k,3])), data.iloc[k,11], 2)
                    
                    cv2.putText(frame, data.iloc[k,-2]+str(np.round(preds,1)), (int(data.iloc[k,0])-5, int(data.iloc[k,1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, data.iloc[k,-1], 2,
                                lineType=cv2.LINE_AA)
                    
                    distance.append(preds)
            
            # 최소 거리 뽑아서 속도 그 차량으로 하기
            end = time.time() # 시간 측정 끝
            vel_time = end - start
            
            # Calculate velocity and print warning message if the velocity high or the distance between car very close.
            '''
            if len(distance) > 0:
            
                current = min(distance) - 1.5 # 1.3은 차의 전장 거리
                if count > 0:
                    #print(vel_time, current, prev)
                    speed = speed_estimate(prev, current, vel_time)
                    speed = round(speed,2)
                    odd_process(current, speed)
                    print('Speed:',speed,'\t','distance:', np.round(current,2))
                
                # 업데이트
                prev = current
                count += 1
            '''
                    
            # 인식되는 차로를 1차선으로 제한하기
            cv2.line(frame,  (500,0), (500,1000), (124, 252, 0))
            cv2.line(frame,  (742,0), (742,1000), (124, 252, 0))
            
            cv2.imshow('video1', frame)
            torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
            
            # Save Video
            #out.write(frame) # 실험 때는 제거
            #print(f"{end - start:.5f} sec") # each frame:
            

        else:
            print("프레임을 받을 수 없습니다.")
            #warn1.speak()
            break
        
          
        
else:
    print('파일을 열 수 없습니다')
    #warn1.speak()

    
# OpenCV 중지
cap.release()
#out.release() # 이것도 실험 때는 제거
cv2.destroyAllWindows()   