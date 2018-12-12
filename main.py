
#%%
######################### EXECUTION NOTICE ################################
# In order to create the data_set 'bird_dataset_crop',
# run the python notebook called : 'preprocessing.ipynb'
# this notebook will create a directory where the images are cropped
# around the bounding boxes of the birds


#%%
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.optim import lr_scheduler



# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset_crop', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)




# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)



# Data initialization and loading
from data import data_transform_train
from data import data_transform_val

    
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transform_train),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transform_val),
    batch_size=args.batch_size, shuffle=False, num_workers=1)    
    




from model import Resnet
from model import Densenet


## we call our model model0 to avoid renaming model

model0 = Densenet()

if use_cuda:
    print('Using GPU')
    model0.cuda()
else:
    print('Using CPU')


## We can change the optimizer to the Adam optimizer which is faster and more accurate
#optimizer = optim.SGD(model0.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = 0.0001)
optimizer = optim.Adam(model0.parameters(), lr=args.lr)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
## adding weight_decay


def train(epoch):
    model0.train()
    #scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # add the auxilary value for inception
        
        output = model0(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()        
        optimizer.step()
        
        #print(len(train_loader.dataset))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model0.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # add the auxilary value for inception
        output= model0(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model0.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
