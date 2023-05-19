import os
import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from . import MyDataset,Utils
from .efficientnet_v2 import efficientnetv2_s

class Recognition():
    def __init__(self,model_path,index_path):
        self.model_path = model_path
        self.index_path = index_path
    def recognise(self,imgs):
        num_classes = 3926
        img_size = 32
        model = efficientnetv2_s(num_classes=num_classes)
        model.eval()

        print('Loading weights from checkpoint (' + self.model_path + ')')
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        ret_chars = []
        transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        output = model(torch.stack([transform(img) for img in imgs]))

        with open(self.index_path,'r',encoding='utf-8') as f:
            label2cha_dict = json.load(f)
        _, predict = torch.max(output.data, 1)
        ret_chars = [label2cha_dict[str(out.item())] for out in predict]
        return ret_chars
    