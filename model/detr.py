# -*- coding: utf-8 -*-
"""
Created on Tue Apr 5 03:56:54 2022
@author: ODD_Team

references: https://github.com/facebookresearch/detr 
"""
import torch
from torchvision import transforms



###############################################################################
# DETR 모델 불러오기 import DETR
class DETR():
    def __init__(self, model_path, backbone):
    # 2. Make COCO dataset
    # COCO classes (91개)
       self.CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
       
       # set colors for visualizations!
       self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], 
                       [0,0,1], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        
        # 3. Preprocessing & bounding box
        # standard PyTorch mean-std input image normalization
       self.transform = transforms.Compose([
            #transforms.Resize((375, 1242)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
       
       # Import model
       # DETR Resnet50 or DETR Resnet101   
       self.model = torch.hub.load(model_path, backbone, pretrained=True)
       self.model.eval()
       
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1).to(self.device)
    
    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(self.device)
        return b
    
    # 4. Detect and return the bounding box
    def detect(self, im):
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0).to(self.device)
    
        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    
        # propagate through the model
        outputs = self.model(img)
    
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
    
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        return probas[keep], bboxes_scaled