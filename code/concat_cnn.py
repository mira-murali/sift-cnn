import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import sampler
from PIL import Image
import imageio
import scipy.io
import numpy as np
import os, sys
import argparse
import time
import pickle
from collections import OrderedDict
from features import get_histogram_sift
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
parser.add_argument('--flat-shape', default=64, type=int, help='linear layer shape')
parser.add_argument('--freezeConv', default=1, type=int, help='set to 0 if conv layers need to be finetuned')
args = parser.parse_args()


current_path= os.path.realpath('mini_resnet.py')
os.environ['CURRENT'] = current_path[:current_path.find('mini')]

if not os.path.isdir(os.path.join(os.environ['CURRENT'], args.foldername)):
    os.mkdir(os.path.join(os.environ['CURRENT'], args.foldername))

current_folder = os.path.join(os.environ['CURRENT'], args.foldername)
outfile = open(os.path.join(current_folder, 'output.txt'), 'w')

sift_bow = pickle.load(open('sift_kmeans_obj_16_cifar_1', 'rb'))
# surf_bow = pickle.load(open('surf_kmeans_obj', 'rb'))
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.training:
            out = self.dropout(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, flat_shape=256, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(flat_shape * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_mini(flat_shape=256, num_classes=10):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2], flat_shape, num_classes)
    return model


class concatCNN(nn.Module):
    def __init__(self, orig_model, num_features=96, num_classes=10):
        super(concatCNN, self).__init__()
        self.nfeatures = num_features
        self.nclasses = num_classes
        self.conv_features = nn.Sequential(*list(orig_model.children())[:-1])

        self.concat_classifier = nn.Sequential(nn.Linear(self.nfeatures, self.nclasses))

        self.conv_features.apply(self.init_weight)
        self.concat_classifier.apply(self.init_weight)
    
    def forward(self, x, hcf):
        conv_features = self.conv_features(x)
        flat_feat = conv_features.view(conv_features.size(0), -1)
        concat_features = torch.cat((flat_feat, hcf), dim=1)
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

    orig_model = resnet_mini(flat_shape=args.flat_shape, num_classes=10)
    base_checkpoint = torch.load(os.path.join(current_folder, 'baseline.pth.tar'))
    orig_model.load_state_dict(base_checkpoint['state_dict'])
    orig_model.to(device)  
    
    model = concatCNN(orig_model, num_features=80)
    model.to(device)

    if args.freezeConv:
        child_counter = 0
        for child in model.children():
            if child_counter < 1:
                for param in child.parameters():
                    param.requires_grad = False
            
            child_counter += 1
    # else:
    #     checkpoint = torch.load(os.path.join(args.load_folder, 'model_best.pth.tar'))
    #     model.load_state_dict(checkpoint['state_dict'])
    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)
    # define transformations

    # resize = transforms.Resize((128, 128))
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
    normal = transforms.Compose([transforms.ToTensor(),
                                 normalize])
    # transform_normalize = T.Compose([
    # T.ToTensor(),
    # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # load data
    train_file = args.train_file
    test_file = args.test_file

    # train_data = ImageDataset(file_name=train_file, transforms=normal)
    # test_data = ImageDataset(file_name=test_file, transforms=normal)
    NUM_TRAIN = 45000
    NUM_VAL = 5000
    train_data = datasets.CIFAR10('../cifar10', train=True, transform=transforms.Compose([transform_augment, normal]), download=True)
    val_data = datasets.CIFAR10('../cifar10', train=True, transform=normal, download=True)
    test_data = datasets.CIFAR10('../cifar10', train=False, transform=normal, download=True)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=ChunkSampler(NUM_TRAIN), num_workers=2)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN), num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=2)

    

    #
    # train model
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    best_acc = 0
    for epoch in range(args.epochs):
        if not args.eval:
            if args.finetune:
                checkpoint = torch.load(os.path.join(os.environ['CURRENT'], args.load_folder))
                model.load_state_dict(checkpoint['state_dict'])
            loss, acc = train(model, criterion, optimizer, train_loader, epoch)
            total_train_acc.append(acc)
            total_train_loss.append(loss)
        # if (epoch+1)%3==0:
            # checkpoint = torch.load(os.path.join(current_folder, 'baseline.pth.tar'))
            # model.load_state_dict(checkpoint['state_dict'])
        val_loss, val_acc = validate(model, criterion, val_loader, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {'state_dict': model.state_dict()}
            torch.save(checkpoint, os.path.join(current_folder, 'model_best.pth.tar'))
        total_val_loss.append(val_loss)
        total_val_acc.append(val_acc)
        # print('Total Test Accuracy: {:.3f}'.format(test_acc))
        # print('Total Test Loss:{:.3f} '.format(test_loss))
        # outfile.write('Total Test Accuracy: {:.3f}'.format(test_acc))
        # outfile.write('Total Test Loss:{:.3f} '.format(test_loss))
        # outfile.write('\n')
        scheduler.step()

    checkpoint = torch.load(os.path.join(current_folder, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_acc = test(model, criterion, test_loader)
    print('Total Test Accuracy: {:.3f}'.format(test_acc))
    print('Total Test Loss:{:.3f} '.format(test_loss))
    outfile.write('Total Test Accuracy: {:.3f}'.format(test_acc))
    outfile.write('Total Test Loss:{:.3f} '.format(test_loss))
    outfile.write('\n')
    np.save(os.path.join(current_folder, 'concat_train_loss.npy'), np.asarray(total_train_loss))
    np.save(os.path.join(current_folder, 'concat_train_acc.npy'), np.asarray(total_train_acc))
    np.save(os.path.join(current_folder, 'concat_val_loss.npy'), np.asarray(total_val_loss))
    np.save(os.path.join(current_folder, 'concat_val_acc.npy'), np.asarray(total_val_acc))




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
            if hist.sum() == 0:
                hist = np.zeros_like(hist)
            else:
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

def validate(model, criterion, val_loader, epoch):
    model.eval()
    start = time.time()
    avg_loss = 0
    total_correct = 0
    total = 0
    print('Validation: ')
    outfile.write('Validation: \n')
    for i, (input, target) in enumerate(val_loader):
        data_time = time.time() - start
        sift_features = []
        for b in range(input.shape[0]):
            numpy_image = input[b].permute(1, 2, 0).numpy()
            numpy_image = (numpy_image - numpy_image.min())/(numpy_image.max()- numpy_image.min())
            numpy_image = (numpy_image*255).astype(np.uint8)
            hist, edges = get_histogram_sift(numpy_image, sift_bow)
            if hist.sum() == 0:
                hist = np.zeros_like(hist)
            else:
                hist = hist/hist.sum()
            sift_features.append(torch.Tensor(hist.reshape(1, -1).tolist()))
        sift_features = torch.cat(sift_features, dim=0).to(device)
        input, target = input.to(device), target.to(device)
        output = model(input, sift_features)
        _, pred = output.max(1)
        loss = criterion(output, target)
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
                  'Batch Time: {:.3f}'.format(epoch, i, len(val_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Accuracy: {:.3f}\t'
                  'Data time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(val_loader), \
                  avg_loss/(i+1), total_correct/total, data_time, batch_time))
            outfile.write('\n')
            start = time.time()
    return avg_loss/len(val_loader), total_correct/total

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
            if hist.sum() == 0:
                hist = np.zeros_like(hist)
            else:
                hist = hist/hist.sum()
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