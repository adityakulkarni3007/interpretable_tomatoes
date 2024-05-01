import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        in_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet(5)
    print(model)

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        
        self.conv3 = ConvBlock(64, 128, pool=True) 
        self.conv4 = ConvBlock(128, 256, pool=True)
        #self.conv5 = ConvBlock(256, 256, pool=True)
        #self.conv6 = ConvBlock(256, 512, pool=True)
        #self.conv7 = ConvBlock(512, 512, pool=True)
        
        self.res2 = nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256))
        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                       nn.Flatten(),
                                       nn.Linear(256, num_diseases))
        
    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        #out = self.conv5(out)
        #out = self.conv6(out)
        #out = self.conv7(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
