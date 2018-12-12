import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
nclasses = 20 

#alexNet not trainable on a CPU
class AlexNet(nn.Module):

    def __init__(self, num_classes=20):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    
# basic network    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = models.resnet152(pretrained=True)                                  
        for params in self.model.parameters():
                params.requires_grad = False            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.1),nn.Linear(num_ftrs, nclasses))
    def forward(self,x):
        return(self.model.forward(x))



class Densenet(nn.Module):
    def __init__(self):
        super(Densenet, self).__init__()
        self.model = models.densenet201(pretrained=True)                                  
        for params in list(self.model.parameters())[0:100]:
                params.requires_grad = False            
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(num_ftrs, nclasses))
    def forward(self,x):
        return(self.model.forward(x))


