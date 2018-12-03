import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import imageio

import scipy.io
import numpy as np
import os, sys
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser('Mini ResNet Training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train model')
    parser.add_argument('--train-file', default='../lists/train_list.mat', type=str, help='Path to train file')
    parser.add_argument('--test-file', default='../lists/test_list.mat', type=str, help='Path to test file')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size while training')
    parser.add_argument('--eval', default=0, type=int, help='Set to 1 if model must only be evaluated')
    parser.add_argument('--finetune', default=0, type=int, help='Set to 1 if model must be further trained')
    parser.add_argument('--load-folder', default='', type=str, help='Path to load saved model (required when finetune is 1)')
    parser.add_argument('--foldername', default='test_run', type=str, help='Folder name (will be created if it does not exist) to store model, plots, etc.')
    parser.add_argument('--print-freq', default=50, type=int, help='how often to print model performance per epoch (in terms of iterations)')
    return parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, file_name, transforms):
        self.data_file = file_name
        self.data = scipy.io.loadmat(self.data_file)
        self.input = self.data['file_list']
        self.label = self.data['labels']
        self.transforms = transforms

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        print('hello')
        print(self.input[idx, 0][0])
        quit()
        input = imageio.imread(self.input[idx, 0][0])
        input = self.transforms(Image.fromarray(input, mode='RGB'))
        label = self.label[idx][0] - 1
        return input, label

class resBlock(nn.Module):
    def __init__(self, num_features):
        super(resBlock, self).__init__()
        self.nfeatures = num_features
        self.block = nn.Sequential(nn.Conv2d(self.nfeatures, self.nfeatures, 3),
                                   nn.BatchNorm2d(),
                                   nn.ReLU(),
                                   nn.Conv2d(self.nfeatures, self.nfeatures, 3)
                                   )

    def forward(self, x):
        y = self.block(x)
        y = y + x
        return y

class miniResnet(nn.Module):
    def __init__(self, input_size=3, output_size=32, flat_shape=1000, num_classes=120):
        super(miniResnet, self).__init__()
        self.in_channels = input_size
        self.out_channels = output_size
        self.nclasses = num_classes
        self.flat_shape = flat_shape
        self.features = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 5),
                                      nn.BatchNorm2d(),
                                      nn.ReLU(),
                                      resBlock(self.out_channels),
                                      nn.BatchNorm2d(),
                                      nn.ReLU(),
                                      nn.Conv2d(self.out_channels, self.out_channels*2, 5),
                                      nn.BatchNorm2d(),
                                      nn.ReLU(),
                                      resBlock(self.out_channels*2),
                                      nn.BatchNorm2d(),
                                      nn.ReLU(),
                                      nn.Conv2d(self.out_channels*2, self.out_channels, 5),
                                      resBlock(self.out_channels),
                                      nn.BatchNorm2d(),
                                      nn.ReLU(),
                                      nn.AvgPool2d())

        self.classifier = nn.Sequential(nn.Linear(self.flat_shape, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, self.nclasses))

    def forward(self, x):
        features = self.features(x)
        flat_feat = features.view(features.size(0), -1)
        print(flat_feat.shape)
        y = self.classifier(flat_feat)
        return y


def main(args):

    args = parse_arguments()
    current_path= os.path.realpath('mini_resnet.py')
    os.environ['CURRENT'] = current_path[:current_path.find('mini')]

    if not os.path.isdir(os.path.join(os.environ['CURRENT'], args.foldername)):
        os.mkdir(os.path.join(os.environ['CURRENT'], args.foldername))

    current_folder = os.path.join(os.environ['CURRENT'], args.foldername)
    outfile = open(os.path.join(current_folder, 'output.txt'), 'w')

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(device)
    # create model
    print('Enter main')
    quit()
    model = miniResnet(input_size=3, output_size=32, flat_shape=5000, num_classes=120)
    model.to(device)
    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # define transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normal = transforms.Compose([transforms.ToTensor(),
                                 normalize])

    # load data
    train_file = args.train_file
    test_file = args.test_file

    train_data = ImageDataset(file_name=train_file, transforms=normal)
    test_file = ImageDataset(file_name=test_file, transforms=normal)

    train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #
    # train model
    best_acc = 0
    for epoch in range(args.epochs):
        if not args.eval:
            if args.finetune:
                checkpoint = torch.load(os.path.join(os.environ['CURRENT'], args.load_folder))
                model.load_state_dict(checkpoint['state_dict'])
            loss, acc = train(model, criterion, optimizer, train_loader, epoch)
            if acc > best_acc:
                best_acc = acc
                checkpoint = {'state_dict': model.state_dict()}
                torch.save(os.path.join(current_folder, 'model_best.pth.tar'), checkpoint)

        checkpoint = torch.load(os.path.join(current_folder, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc = test(model, criterion, test_loader)
        print('Total Test Accuracy: {:.3f}'.format(test_acc))
        print('Total Test Loss:{:.3f} '.format(test_loss))
        outfile.write('Total Test Accuracy: {:.3f}'.format(test_acc))
        outfile.write('Total Test Loss:{:.3f} '.format(test_loss))
        outfile.write('\n')

def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    start = time.time()
    avg_loss = 0
    total_correct = 0
    total = 0
    print('Training: ')
    outfile.write('Training: \n')
    for i, (input, target) in enumerate(train_loader):
        data_time = time.time() - start
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, pred = output.max(1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        total += target.size(0)
        correct = (pred==target).sum().item()
        total_correct += correct
        batch_time = time.time() - start
        if (i+1)%args.print_freq==0:
            print('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Accuracy: {:.3f}\t'
                  'Data time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(train_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Accuracy: {:.3f}\t'
                  'Data time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(train_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('\n')

    return avg_loss/len(train_loader), total_correct/total

def test(model, criterion, test_loader):
    model.eval()
    total_correct = 0
    total = 0
    avg_loss = 0
    start = time.time()
    print('Testing: ')
    outfile.write('Testing: \n')
    for i, (input, target) in enumerate(test_loader):
        data_time = time.time() - start
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, pred = output.max(1)
        loss = criterion(output, target)
        avg_loss += loss.item()
        correct = (pred==target).sum().item()
        total_correct += correct
        total += target.size(0)

        if (i+1)%args.print_freq==0:
            print('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Accuracy: {:.3f}\t'
                  'Data time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(test_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Accuracy: {:.3f}\t'
                  'Data time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(test_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('\n')

    return avg_loss/len(test_loader), total_correct/total

if __name__ == '__main___':
    print('hello')
    main(sys.argv)