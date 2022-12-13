#import torch.nn
from torch.nn import Sequential, Module, Conv2d, Linear, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, batch_norm

# https://www.researchgate.net/publication/317425461_Training_Quantized_Nets_A_Deeper_Understanding
class VGG9(Module):
    def __init__(self):
        super(VGG9, self).__init__()
        self.maxpooling = MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        self.conv5 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv6 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.fc7 = Linear(in_features=8192, out_features=1024)
        self.fc8 = Linear(in_features=1024, out_features=1024)
        self.fc9 = Linear(in_features=1024, out_features=10)

        self.bn128 = BatchNorm2d(128)
        self.bn256 = BatchNorm2d(256)
        self.bn512 = BatchNorm2d(512)


    def forward(self, x):
        x = self.bn128(relu(self.conv1(x)))
        x = self.bn128(relu(self.conv2(x)))
        x = self.maxpooling(x)

        x = self.bn256(relu(self.conv3(x)))
        x = self.bn256(relu(self.conv4(x)))
        x = self.maxpooling(x)

        x = self.bn512(relu(self.conv5(x)))
        x = self.bn512(relu(self.conv6(x)))
        x = self.maxpooling(x)

        # flatten
        x = x.reshape(x.shape[0], -1)
        x = relu(self.fc7(x))
        x = relu(self.fc8(x))
        x = relu(self.fc9(x))

        return x




class VGG16(Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.maxpooling = MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.fc14 = Linear(in_features=512, out_features=4096)
        self.fc15 = Linear(in_features=4096, out_features=4096)
        self.fc16 = Linear(in_features=4096, out_features=10)


        

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.maxpooling(x)

        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = self.maxpooling(x)

        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = relu(self.conv7(x))
        x = self.maxpooling(x)

        x = relu(self.conv8(x))
        x = relu(self.conv9(x))
        x = relu(self.conv10(x))
        x = self.maxpooling(x)

        x = relu(self.conv11(x))
        x = relu(self.conv12(x))
        x = relu(self.conv13(x))
        x = self.maxpooling(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        x = relu(self.fc14(x))
        x = relu(self.fc15(x))
        x = relu(self.fc16(x))

        return x


