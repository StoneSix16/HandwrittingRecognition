import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import json
from torchvision import transforms
from PIL import Image
import cv2
from efficientnet_v2 import efficientnetv2_s

class Recognition():
    def __init__(self,model_path,index_path):
        self.model_path = model_path
        self.label2cha_dict = {}
        with open(index_path,'r',encoding='utf-8') as f:
            self.label2cha_dict = json.load(f)
    def recognise(self,raw_imgs):
        # init params
        num_classes = 3926
        img_size = 32
        model = efficientnetv2_s(num_classes=num_classes)
        model.eval()

        # load model
        print('Loading weights from checkpoint (' + self.model_path + ')')
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # preprocess
        transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transback = transforms.ToPILImage(mode='RGB') 

        imgs = []
        for i,img in enumerate(raw_imgs):
            grayimg = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)
            _,newimg = cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY)
            img = Image.fromarray(newimg).convert('RGB')
            imgs.append(transform(img))

        output = model(torch.stack(imgs))
        _, predict = torch.topk(output.data, 5, 1)
        chars = []
        for line in predict:
            chars.append([self.label2cha_dict[str(i.item())] for i in line])
        return [transback(img) for img in imgs],chars
    