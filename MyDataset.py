import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, index_path, num_class, transforms=None):
        super(MyDataset,self).__init__()
        images = []
        labels = []
        with open(index_path, 'r') as f:
            for line in f:
                [index,addr] = line.split(':')
                if int(index) >= num_class:
                    break
                images.append(addr.strip('\n'))
                labels.append(int(index))
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
    
#为每个类映射索引并提供图片路径

def classes_index(root, out_path, num_class=None):
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)
 
    with open(out_path, 'w') as f:
        cnt = 0
        for dir in dirs:
            files = os.listdir(os.path.join(root, dir).replace('\\','/'))
            for file in files:
                addr = os.path.join(root, dir, file).replace('\\','/')
                f.write(f"{cnt}:{addr}" + '\n')
            cnt+=1