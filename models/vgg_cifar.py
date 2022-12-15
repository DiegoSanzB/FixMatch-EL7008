from torch.nn import Dropout, Sequential, Module, Conv2d, Linear, MaxPool2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d

'''
https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/vgg.py
'''

NCLASSES = 10

class VGG11_BN(Module):
    def __init__(self):
        super(VGG11_BN, self).__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = AdaptiveAvgPool2d((1, 1))

        self.classifier = Sequential(
            Linear(512 * 1 * 1, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, NCLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

