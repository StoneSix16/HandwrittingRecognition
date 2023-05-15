import os
from PIL import Image
from torch.utils.data import Dataset
import json
import struct

class MyDataset(Dataset):
    def __init__(self, data_path, num_class, transforms=None):
        super(MyDataset,self).__init__()
        images = []
        labels = []
        dirs = os.listdir(data_path)
        for dir in dirs:
            files = os.listdir(os.path.join(data_path, dir).replace('\\','/'))
            for file in files:
                addr = os.path.join(data_path, dir, file).replace('\\','/')
                images.append(addr)
                labels.append(int(dir))
        self.images = images
        self.labels = labels
        self.transforms = transforms
    def __getitem__(self, index):
        try:
            image = Image.open(self.images[index]).convert('RGB')
            label = self.labels[index]
            if self.transforms is not None:
                image = self.transforms(image)
            return image,label
        except Exception as e:
            print(e)
            return self.images[0], self.labels[0]

    def __len__(self):
        return len(self.labels)
        