"""
    @ Author_odd team
    @ this program is design by ODD_team.
    @ using open-cv, we will predict distance & and give the warning message.
    @ If we have the chance imporve it by application, we will use GPS API to estimate accurate system
"""

import os
import pandas as pd
import numpy as np
import pickle
import time
import torch
import cv2
#import threading
from model.detr import DETR
from model.glpdepth import GLP
from model.ODD import ODD
import xgboost as xgb
import warnings
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################# Start ###########################
warnings.filterwarnings(action='ignore')

'''
function define
'''
'''
# sound1
import pyttsx3 
s = pyttsx3.init()
speak = "전방을 주의하세요"

# Speaking function
def speaking():
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
            beepsound()
          
    elif speed>=40:
        if zloc<15:
            print('실행 40')            
            beepsound()

    elif speed>=10:
        if zloc<7:     
            print('실행 10')
            beepsound()
            
    else:
        pass
'''
                
'''
Model 및 카메라 정의
'''
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

# ODD process 불러오기
ODD_process = ODD()

# Z-estimator 불러오기
'''
사용한 변수
: xmin, ymin, xmax, ymax, depth_mean, depth_median, depth_max, depth_mean_trim, width, height, Misc, bicycle, car, person, train, truck

'''
z_model = pickle.load(open('./weights/xgb_model.model', 'rb'))

#스케일러 불러오기
scaler = pickle.load(open('./weights/standard_scaler.pkl', 'rb'))
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
            scores, boxes = DETR.detect(pil_image) # Detection
            
            
            '''
            Step2) apply GLP_Depth 
            '''
            # Make depth map
            prediction = GLPdepth.predict(pil_image, img_shape)
            
            
            '''
            Step3) apply z-loc estimator
            '''
            data = ODD_process.make_dataset_from_pretrained_model(scores, boxes, prediction, DETR) # dataset 만들기
            
            # input text and draw bbox
            distance = []
            for k in data.index:
                x_range = np.arange(int(data.iloc[k,0]), int(data.iloc[k,2])+1) # xmax~xmin 
                line_range = np.arange(500, 742+1)
                
                # if images are closed and mixed, apply it
                if len(np.intersect1d(x_range, line_range)) >= 10: 
                    classes = data.iloc[k,-2] # class info
                    '''
                    Z-model > in this case we will use xgboost model
                    '''
                    #Misc, bicycle, car, person, train, truck
                    if classes == 'Misc':
                        array = torch.tensor([[1,0,0,0,0,0]])
                    elif classes == 'bicycle':
                        array = torch.tensor([[0,1,0,0,0,0]])
                    elif classes == 'car':
                        array = torch.tensor([[0,0,1,0,0,0]])
                    elif classes == 'person':
                        array = torch.tensor([[0,0,0,1,0,0]])
                    elif classes == 'train':
                        array = torch.tensor([[0,0,0,0,1,0]])
                    elif classes == 'truck':
                        array = torch.tensor([[0,0,0,0,0,1]])
                    #input_data = torch.tensor([[x1,y1,x2,y2,depth_mean,depth_median, depth_max, depth_mean_trim, width, height]])
                    #input_data_scaler = torch.tensor(scaler.transform(input_data)) # scaler 적용
                    input_data_scaler = torch.tensor(scaler.transform(data.iloc[[k],0:10]))
                    
                    model_data = torch.cat([input_data_scaler, array], dim=1)
                    dataframe = pd.DataFrame(model_data,columns=[0,1,2,3,4,5,6,7,8,9,'Misc','bicycle','car','person','train','truck'])
                    
                    # Predict
                    d_test=xgb.DMatrix(data=dataframe)
                    preds = z_model.predict(d_test)
                    
                    # error1: 좌표는 int형.
                    cv2.rectangle(frame, (int(data.iloc[k,0]), int(data.iloc[k,1])), (int(data.iloc[k,2]), int(data.iloc[k,3])), data.iloc[k,11], 2)
                    
                    cv2.putText(frame, data.iloc[k,-2]+str(np.round(preds,1)), (int(data.iloc[k,0])-5, int(data.iloc[k,1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, data.iloc[k,-1], 2,
                                lineType=cv2.LINE_AA)
                    
                    if classes not in ['Misc','person']:
                        distance.append(preds)
        
                    
            # 인식되는 차로를 1차선으로 제한하기
            cv2.line(frame,  (500,0), (500,1000), (124, 252, 0))
            cv2.line(frame,  (742,0), (742,1000), (124, 252, 0))
            
            # 최소 거리 뽑아서 속도 그 차량으로 하기
            end = time.time() # 시간 측정 끝
            vel_time = end - start
            
            # Calculate velocity and print warning message if the velocity high or the distance between car very close.
            '''
            if len(distance) > 0:
            
                current = min(distance) - 1.5 # 1.3은 차의 전장 거리
                if count > 0:
                    speed = speed_estimate(prev, current, vel_time)
                    speed = round(speed,2)
                    odd_process(current, speed)
                    print('Speed:',speed,'\t','distance:', np.round(current,2))
                
                
                # 업데이트
                prev = current
                count += 1
            '''
            
            cv2.imshow('video1', frame)
            
            # Save Video
            #out.write(frame) # 실험 때는 제거
            #print(f"{end - start:.5f} sec") # each frame:
            torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
            
            # 말하기 (Multi-thread 이용)
            #beepsound()
            
        else:
            print("프레임을 받을 수 없습니다.")
            #warn1.speak()
            break
        
          
        #error message
else:
    print('파일을 열 수 없습니다')
    #warn1.speak()

    
# OpenCV 중지
cap.release()
#out.release() # 이것도 실험 때는 제거 -> it helps release the memory in your enviroment.
cv2.destroyAllWindows()   

