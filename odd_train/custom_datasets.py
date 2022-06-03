# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:24:04 2022

@author: ODD
@preparing data set
# reference: https://velog.io/@devlee247/Pytorch%EC%9D%98-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B2%98%EB%A6%AC-Dataset-Dataloader
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# get data set
class CustomDataset():
    def __init__(self, data, variable, scaler=False, train=False, onehot=True):
        
        self.df=data
        
        # One-hot
        if onehot==True:
            self.inp = self.df[[col for col in self.df[variable].columns if col not in ['Misc', 'bicycle', 'car', 'people', 'train', 'truck']]].values
            self.onehot = self.df[[col for col in self.df[variable].columns if col in ['Misc', 'bicycle', 'car', 'people', 'train', 'truck']]].values
        else:
            self.inp = self.df[variable].values
        self.outp = self.df[['zloc']].values # zloc
        
        #self.scaler = MinMaxScaler()
        if train==True:
            self.scaler = StandardScaler().fit(self.inp)
        else: 
            self.scaler = train
        
        # Scaler
        if scaler==True:
            self.inp = self.scaler.transform(self.inp)
            if onehot==True:
                self.inp = np.concatenate([self.onehot, self.inp], axis=1)
                
                
	
    def __len__(self):
		# return the length of dataset
        return len(self.inp) # 1314
    
    def __getitem__(self,idx):
        inp = torch.FloatTensor(self.inp[idx])
        outp = torch.FloatTensor(self.outp[idx])
        return inp, outp # return the according to idx, return the input and output.