# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:55:50 2022

@author: ODD_team
@based on LSTM model 

# Make ourself
"""

import torch
import torch.nn as nn

# z-location estimator
class Zloc_Estimaotor(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super().__init__()
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        #set Layer
        layersize=[306, 154, 76]
        layerlist= []
        n_in=hidden_dim
        for i in layersize:
            layerlist.append(nn.Linear(n_in,i))
            layerlist.append(nn.ReLU())
            #layerlist.append(nn.BatchNorm1d(i))
            #layerlist.append(nn.Dropout(0.1))
            n_in=i           
        layerlist.append(nn.Linear(layersize[-1],1))
        #layerlist.append(nn.Sigmoid())
        
        self.fc=nn.Sequential(*layerlist)

        
    def forward(self, x):
        out, hn = self.rnn(x)
        output = self.fc(out[:,-1])
        return output
    
#class of LSTM
class LSTM():
    #base-line of zloc model
    def __init__(self, path):
        self.input_dim = 16 # 15
        self.hidden_dim = 612
        self.layer_dim = 3
        
        self.model = Zloc_Estimaotor(self.input_dim, self.hidden_dim, self.layer_dim)
        self.model.load_state_dict(torch.load(path))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cuda'
        
    # predict the zlocation and return it
    def predict(self, data):
        self.zloc = self.model(data.reshape(-1,1, self.input_dim).to(self.device))
        return self.zloc.cpu()