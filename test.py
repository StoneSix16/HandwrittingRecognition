import os
import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from MyDataset import MyDataset
import Utils
from efficientnet_v2 import efficientnetv2_s
import argparse # 提取命令行参数

parser = argparse.ArgumentParser(description='EfficientNetV2 arguments')
parser.add_argument('--mode', dest='mode', type=str, default='train', help='Mode of net')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='Epoch number of training')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512, help='Value of batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='Value of lr')
parser.add_argument('--img_size', dest='img_size', type=int, default=32, help='reSize of input image')
parser.add_argument('--data_root', dest='data_root', type=str, default='./data/', help='Path to data')
parser.add_argument('--log_root', dest='log_root', type=str, default='./log/', help='Path to model.pth')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=3926, help='Classes of character')
parser.add_argument('--index_path', dest='index_path', type=str, default='./label2cha.json', help='Path to index.json')
parser.add_argument('--model_path', dest='model_path', type=str, default='./efficientnet_45.pth', help='model for test')
parser.add_argument('--img_path', dest='img_path', type=str, default='./asserts/wen.png', help='Path to demo image')
args = parser.parse_args(namespace=argparse.Namespace())

def train(args):
    print("===Train EffNetV2===")
    # 归一化处理，不一定要这样做，看自己的需求，只是预训练模型的训练是这样设置的
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.ColorJitter()])  
 
    train_set = MyDataset(args.data_root, num_class=args.num_classes, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda:0')
    # 加载模型
    model = efficientnetv2_s(num_classes=args.num_classes)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调整函数，在训练效果下降时，会降低学习率以进一步微调模型
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)
    print("load model...")
    
    # 加载最近保存了的参数
    if Utils.has_log_file(args.log_root):
        max_log = Utils.find_max_log(args.log_root)
        print("continue training with " + max_log + "...")
        checkpoint = torch.load(max_log)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    else:
        print("train for the first time...")
        loss = 0.0
        epoch = 0
 
    while epoch < args.epoch:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('epoch %5d: batch: %5d, loss: %8f, lr: %f' % (
                    epoch + 1, i + 1, running_loss / 200, optimizer.state_dict()['param_groups'][0]['lr']))
                running_loss = 0.0
 
        scheduler.step(loss)
        # 每个epoch结束后就保存最新的参数
        print('Save checkpoint...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   args.log_root + 'log' + str(epoch) + '.pth')
        print('Saved')
        epoch += 1
 
    print('Finish training')

def evaluate(args):
    print("===Evaluate EffNetV2===")
    # 在图像大小和正则化方面与训练集保持一致
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
 
    model = efficientnetv2_s(num_classes=args.num_classes)
    model.eval()
    if Utils.has_log_file(args.log_root):
        file = Utils.find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No log file")
 
    model.to(torch.device('cuda:0'))
    test_loader = DataLoader(MyDataset(args.data_root, num_class=args.num_classes, transforms=transform),batch_size=args.batch_size, shuffle=False)
    total = 0.0
    correct = 0.0
    #total&correct for a group of batches
    _total = 0.0
    _correct = 0.0
    print("Evaluating...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            _total += labels.size(0)
            _correct += (predict == labels).sum().item()
            total += _total
            correct += _correct
            if i % 20 == 19:
                print(f'total:{_total}, correct:{_correct}, acc:{correct/total}')
                _total = 0.0
                _correct = 0.0
    acc = correct / total * 100
    print(f'acc:{acc}%')

def demo(args, char_dict):
    print('==Demo EfficientNetV2===')
    print('Input Image: ', args.img_path)
    # 这个地方要和train一致，不过colorJitter可有可无
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = Image.open(args.img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0) # 增维
    model = efficientnetv2_s(num_classes=args.num_classes)
    model.eval()
    if Utils.has_log_file(args.log_root):
        file = Utils.find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No log file")
 
    with torch.no_grad():
        output = model(img)
    _, predict = torch.sort(output,descending=True)
    # value, pred = torch.max(output.data, 1)
    # print(value[0,:5],pred[0,:5])
    chas = predict[0,:5].numpy().tolist()
    print(chas)
    print(f'predict:{[char_dict[str(chas[i])] for i in range(len(chas))]}')
    f.close()


if __name__ == '__main__':
    num_dict = {}
    with open(args.index_path,'r',encoding='utf-8') as f:
        num_dict = json.load(f)
 
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'demo':
        demo(args, num_dict)
    else:
        print('Unknown mode')