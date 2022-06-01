# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:08:07 2022

@author: ODD
"""

import pandas as pd
glp_kitti_data = pd.read_csv('./glp_kitti_data_iou.csv')
glp_kitti_data = glp_kitti_data.sample(frac=1, random_state=2022) 
glp_kitti_data.info()

# train: 70%, valid: 15%, test: 15%
kitti_length = len(glp_kitti_data)
train_len = int(kitti_length*0.8)
valid_len = int(kitti_length*0.1)
test_len = int(kitti_length*0.1)

kitti_train = glp_kitti_data.iloc[:train_len,:]
kitti_valid = glp_kitti_data.iloc[train_len:(train_len+valid_len),:]
kitti_test = glp_kitti_data.iloc[(train_len+valid_len):,:]

# width, height 열 추가
kitti_train['width'] = kitti_train['xmax'] - kitti_train['xmin']
kitti_train['height'] = kitti_train['ymax'] - kitti_train['ymin']
kitti_valid['width'] = kitti_valid['xmax'] - kitti_valid['xmin']
kitti_valid['height'] = kitti_valid['ymax'] - kitti_valid['ymin']
kitti_test['width'] = kitti_test['xmax'] - kitti_test['xmin']
kitti_test['height'] = kitti_test['ymax'] - kitti_test['ymin']

#print(kitti_train['class'].value_counts()/len(kitti_train))
#print(kitti_valid['class'].value_counts()/len(kitti_valid))
#print(kitti_test['class'].value_counts()/len(kitti_test))

# 저장
kitti_train.to_csv('./iou_train.csv', index=False)
kitti_valid.to_csv('./iou_valid.csv', index=False)
kitti_test.to_csv('./iou_test.csv', index=False)


'''
# KITTI_VKITTI (7:1.5:1.5)
from sklearn.model_selection import StratifiedShuffleSplit
glp_vkitti_data = pd.read_csv('./glp_vkitti_data.csv')
glp_vkitti_data = glp_vkitti_data[glp_kitti_data.columns]

vkitti_kitti_data = pd.concat([glp_kitti_data, glp_vkitti_data], axis=0)
print(vkitti_kitti_data) # 44171개

vkitti_kitti_data.reset_index(inplace=True)
vkitti_kitti_data.drop('index', axis=1, inplace=True)


# 데이터 추출
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42)
for train_idx, test_idx in split.split(vkitti_kitti_data, vkitti_kitti_data['weather']):
    vkitti_kitti_test = vkitti_kitti_data.loc[test_idx]
    vkitti_kitti_test['width'] = vkitti_kitti_test['xmax'] - vkitti_kitti_test['xmin']
    vkitti_kitti_test['height'] = vkitti_kitti_test['ymax'] - vkitti_kitti_test['ymin']
    
    
    vkitti_kitti_train0 = vkitti_kitti_data.loc[train_idx]
    vkitti_kitti_train0.reset_index(inplace=True)
    vkitti_kitti_train0.drop('index', axis=1, inplace=True)
    
    # train-valid
    split0 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.17648, random_state = 42)
    for train_idx, valid_idx in split0.split(vkitti_kitti_train0, vkitti_kitti_train0['weather']):
        vkitti_kitti_train = vkitti_kitti_train0.loc[train_idx]
        vkitti_kitti_train['width'] = vkitti_kitti_train['xmax'] - vkitti_kitti_train['xmin']
        vkitti_kitti_train['height'] = vkitti_kitti_train['ymax'] - vkitti_kitti_train['ymin']
        
        vkitti_kitti_valid = vkitti_kitti_train0.loc[valid_idx]
        vkitti_kitti_valid['width'] = vkitti_kitti_valid['xmax'] - vkitti_kitti_valid['xmin']
        vkitti_kitti_valid['height'] = vkitti_kitti_valid['ymax'] - vkitti_kitti_valid['ymin']
        
        vkitti_kitti_train.reset_index(inplace=True)
        vkitti_kitti_train.drop('index', axis=1, inplace=True)
        
        vkitti_kitti_valid.reset_index(inplace=True)
        vkitti_kitti_valid.drop('index', axis=1, inplace=True)
        
        
print(len(vkitti_kitti_train)) # 30919
print(len(vkitti_kitti_valid)) # 6626
print(len(vkitti_kitti_test)) # 6626

print(vkitti_kitti_train['weather'].value_counts()/len(vkitti_kitti_train))
print(vkitti_kitti_valid['weather'].value_counts()/len(vkitti_kitti_valid))
print(vkitti_kitti_test['weather'].value_counts()/len(vkitti_kitti_test))

vkitti_kitti_train.to_csv('./vkitti_kitti_train.csv', mode='a', index=False)
vkitti_kitti_valid.to_csv('./vkitti_kitti_valid.csv', mode='a', index=False)
vkitti_kitti_test.to_csv('./vkitti_kitti_test.csv', mode='a', index=False)
'''

'''
glp_vkitti_data = glp_vkitti_data[glp_kitti_data.columns]
vkitti_length = len(glp_vkitti_data)
train_len = int(vkitti_length*0.7)
valid_len = int(vkitti_length*0.15)
test_len = int(vkitti_length*0.15)

vkitti_train = glp_vkitti_data.iloc[:train_len,:]
vkitti_valid = glp_vkitti_data.iloc[train_len:(train_len+valid_len),:]
vkitti_test = glp_vkitti_data.iloc[(train_len+valid_len):,:]

# width, height 열 추가
vkitti_train['width'] = vkitti_train['xmax'] - vkitti_train['xmin']
vkitti_train['height'] = vkitti_train['ymax'] - vkitti_train['ymin']
vkitti_valid['width'] = vkitti_valid['xmax'] - vkitti_valid['xmin']
vkitti_valid['height'] = vkitti_valid['ymax'] - vkitti_valid['ymin']
vkitti_test['width'] = vkitti_test['xmax'] - vkitti_test['xmin']
vkitti_test['height'] = vkitti_test['ymax'] - vkitti_test['ymin']

# VKITTI와 KITTI 
vkitti_train = pd.concat([vkitti_train, kitti_train])
vkitti_valid = pd.concat([vkitti_valid, kitti_valid])
vkitti_test = pd.concat([vkitti_test, kitti_test])

# 저장
vkitti_train.to_csv('./vkitti_kitti_train.csv', mode='a', index=False)
vkitti_valid.to_csv('./vkitti_kitti_valid.csv', mode='a', index=False)
vkitti_test.to_csv('./vkitti_kitti_test.csv', mode='a', index=False)
'''

