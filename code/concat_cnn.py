import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import imageio
import scipy.io
import numpy as np
import os, sys
import argparse
import time
import pickle
from collections import OrderedDict
from features import get_histogram_sift, get_histogram_surf
from multiprocessing import Pool

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)
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
parser.add_argument('--num_workers', default=2, type=int, help='Num workers')
args = parser.parse_args()


current_path= os.path.realpath('mini_resnet.py')
os.environ['CURRENT'] = current_path[:current_path.find('mini')]

if not os.path.isdir(os.path.join(os.environ['CURRENT'], args.foldername)):
    os.mkdir(os.path.join(os.environ['CURRENT'], args.foldername))

current_folder = os.path.join(os.environ['CURRENT'], args.foldername)
outfile = open(os.path.join(current_folder, 'output.txt'), 'w')

sift_bow = pickle.load(open('sift_kmeans_obj', 'rb'))
surf_bow = pickle.load(open('surf_kmeans_obj', 'rb'))
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
        input = imageio.imread('../Images/' + self.input[idx, 0][0])
        input = self.transforms(Image.fromarray(input, mode='RGB'))
        label = self.label[idx][0] - 1
        return input, label

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super(resBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_downsample = is_downsample
        self.block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.LeakyReLU())
        
        self.downsample = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(self.out_channels))
        
        self.block.apply(self.init_weight)
        self.downsample.apply(self.init_weight)
    
    def forward(self, x):
        residual = x
        y = self.block(x)
        if self.is_downsample:
            residual = self.downsample(x)
        y = y + residual
        return y
    
    def init_weight(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)



class miniResnet(nn.Module):
    def __init__(self, input_size=3, output_size=128, flat_shape=1000, num_classes=120):
        super(miniResnet, self).__init__()
        self.in_channels = input_size
        self.out_channels = output_size
        self.nclasses = num_classes
        self.flat_shape = flat_shape
        self.conv_features = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=7, bias=False),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.LeakyReLU(),
                                      nn.MaxPool2d(3, stride=2),
                                      resBlock(self.out_channels, self.out_channels),
                                      resBlock(self.out_channels, self.out_channels),
                                      resBlock(self.out_channels, self.out_channels),
                                    #   nn.MaxPool2d(2, stride=2),
                                      resBlock(self.out_channels, self.out_channels//2, is_downsample=True),
                                      resBlock(self.out_channels//2, self.out_channels//2),
                                      resBlock(self.out_channels//2, self.out_channels//2),
                                    #   nn.MaxPool2d(2, stride=2),
                                      resBlock(self.out_channels//2, self.out_channels//4, is_downsample=True),
                                      resBlock(self.out_channels//4, self.out_channels//4),
                                      resBlock(self.out_channels//4, self.out_channels//4),
                                      nn.AvgPool2d(7, stride=1))

        self.linear_features = nn.Sequential(nn.Linear(self.flat_shape, 512),
                                        nn.Dropout(),
                                        nn.LeakyReLU(), 
                                        )
        
        self.classifier = nn.Sequential(nn.Linear(512, 256),
                                        nn.LeakyReLU(),
                                        nn.Dropout(),
                                        nn.Linear(256, self.nclasses))
    

        self.conv_features.apply(self.init_weight)
        self.linear_features.apply(self.init_weight)
        self.classifier.apply(self.init_weight)
         
    def init_weight(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
        
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)

    def forward(self, x):
        conv_features = self.conv_features(x)
        flat_feat = conv_features.view(conv_features.size(0), -1)
        y = self.linear_features(flat_feat)
        y = self.classifier(y)
        return y

class concatCNN(nn.Module):
    def __init__(self, orig_model, num_features=512, num_classes=10):
        super(concatCNN, self).__init__()
        self.nfeatures = num_features
        self.nclasses = num_classes
        self.conv_features = orig_model.conv_features
        self.linear_features = orig_model.linear_features
        self.projection = nn.Sequential(nn.Linear(self.nfeatures, 128),
                                        nn.LeakyReLU(),
                                        nn.Dropout(),
                                        )
        self.concat_classifier = nn.Sequential(nn.Linear(256, self.nclasses))

        self.conv_features.apply(self.init_weight)
        self.linear_features.apply(self.init_weight)
        self.projection.apply(self.init_weight)
        self.concat_classifier.apply(self.init_weight)
    
    def forward(self, x, hcf):
        conv_features = self.conv_features(x)
        flat_feat = conv_features.view(conv_features.size(0), -1)
        y = self.linear_features(flat_feat)
        proj_out = self.projection(y)
        concat_features = torch.cat((proj_out, hcf), dim=1)
        out = self.concat_classifier(concat_features)
        return out
    
    def init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)

        elif type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

def main():

    # create model

    orig_model = miniResnet(input_size=3, output_size=128, flat_shape=1152, num_classes=10)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     orig_model = nn.DataParallel(orig_model)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    base_checkpoint = torch.load(os.path.join(current_folder, 'baseline.pth.tar'))
    orig_model.load_state_dict(base_checkpoint['state_dict'])
    orig_model.to(device)  
    
    model = concatCNN(orig_model)
    model.to(device)

    child_counter = 0
    for child in model.children():
        if child_counter < 2:
            for param in child.parameters():
                param.requires_grad = False
        
        child_counter += 1
    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    # define transformations

    # resize = transforms.Resize((128, 128))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normal = transforms.Compose([transforms.ToTensor(),
                                 normalize])
    # load data
    train_file = args.train_file
    test_file = args.test_file

    # train_data = ImageDataset(file_name=train_file, transforms=normal)
    # test_data = ImageDataset(file_name=test_file, transforms=normal)

    train_data = datasets.CIFAR10('../cifar10', train=True, transform=normal, download=True)
    test_data = datasets.CIFAR10('../cifar10', train=False, transform=normal, download=True)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    

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
                torch.save(checkpoint, os.path.join(current_folder, 'model_best.pth.tar'))
        if (epoch+1)%3==0:
            # checkpoint = torch.load(os.path.join(current_folder, 'model_best.pth.tar'))
            # model.load_state_dict(checkpoint['state_dict'])
            test_loss, test_acc = test(model, criterion, test_loader)
            print('Total Test Accuracy: {:.3f}'.format(test_acc))
            print('Total Test Loss:{:.3f} '.format(test_loss))
            outfile.write('Total Test Accuracy: {:.3f}'.format(test_acc))
            outfile.write('Total Test Loss:{:.3f} '.format(test_loss))
            outfile.write('\n')
        scheduler.step()

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
        args_list = []
        sift_features = []
        for b in range(input.shape[0]):
            numpy_image = input[b].permute(1, 2, 0).numpy()
            numpy_image = (numpy_image - numpy_image.min())/(numpy_image.max()- numpy_image.min())
            numpy_image = (numpy_image*255).astype(np.uint8)
            hist, edges = get_histogram_sift(numpy_image, sift_bow)
            hist = hist/hist.sum()
            sift_features.append(torch.Tensor(hist.reshape(1, -1).tolist()))

        sift_features = torch.cat(sift_features, dim=0).to(device)
        input, target = input.to(device), target.to(device)
        output = model(input, sift_features)
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
            start = time.time()
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
        args_list = []
        sift_features = []
        for b in range(input.shape[0]):
            numpy_image = input[b].permute(1, 2, 0).numpy()
            numpy_image = (numpy_image - numpy_image.min())/(numpy_image.max()- numpy_image.min())
            numpy_image = (numpy_image*255).astype(np.uint8)
            hist, edges = get_histogram_sift(numpy_image, sift_bow)
            sift_features.append(torch.Tensor(hist.reshape(1, -1).tolist()))

        sift_features = torch.cat(sift_features, dim=0).to(device)
        input, target = input.to(device), target.to(device)
        output = model(input, sift_features)
        _, pred = output.max(1)
        loss = criterion(output, target)
        avg_loss += loss.item()
        correct = (pred==target).sum().item()
        total_correct += correct
        total += target.size(0)

        # if (i+1)%args.print_freq==0:
        #     print('Epoch: [{}][{}/{}]\t'
        #           'Loss: {:.3f}\t'
        #           'Accuracy: {:.3f}\t'
        #           'Data time: {:.3f}\t'
        #           'Batch Time: {:.3f}'.format(epoch, i, len(test_loader), \
        #           avg_loss/(i+1), total_correct/total, data_time, batch_time))
        #     outfile.write('Epoch: [{}][{}/{}]\t'
        #           'Loss: {:.3f}\t'
        #           'Accuracy: {:.3f}\t'
        #           'Data time: {:.3f}\t'
        #           'Batch Time: {:.3f}'.format(epoch, i, len(test_loader), \
        #           avg_loss/(i+1), total_correct/total, data_time, batch_time))
        #     outfile.write('\n')

    return avg_loss/len(test_loader), total_correct/total

if __name__ == '__main__':
    main()