from torch.nn import Module, Conv2d, Linear, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, batch_norm

NCLASSES = 10

class NetCN4(Module):
  def __init__(self):
    super(NetCN4, self).__init__()
    self.nclasses = NCLASSES
    self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = BatchNorm2d(64)
    self.conv2 = Conv2d(64, 64, 3, padding=1)
    self.bn2 = BatchNorm2d(64)
    self.maxp = MaxPool2d(kernel_size=3)
    self.conv3 = Conv2d(64, 64, 3, padding=1)
    self.bn3 = BatchNorm2d(64)
    self.conv4 = Conv2d(64, 64, 3, padding=1)
    self.bn4 = BatchNorm2d(64)
    self.fc_1 = Linear(6400, 64)
    self.fc_last = Linear(64, self.nclasses)

  def forward(self, x):
    x = self.bn1(relu(self.conv1(x)))
    x = self.bn2(relu(self.conv2(x)))
    x = self.maxp(x)
    x = self.bn3(relu(self.conv3(x)))
    x = self.bn4(relu(self.conv4(x)))
    x = x.view(x.size()[0], -1)
    x = relu(self.fc_1(x))
    x = self.fc_last(x)
    return x

class NetCN8(Module):
  def __init__(self):
    super(NetCN8, self).__init__()
    self.nclasses = NCLASSES
    self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = BatchNorm2d(64)
    self.conv2 = Conv2d(64, 64, 3, padding=1)
    self.bn2 = BatchNorm2d(64)
    self.maxp = MaxPool2d(kernel_size=3)

    self.conv3 = Conv2d(64, 64, 3, padding=1)
    self.bn3 = BatchNorm2d(64)
    self.conv4 = Conv2d(64, 64, 3, padding=1)
    self.bn4 = BatchNorm2d(64)
    self.maxp2 = MaxPool2d(kernel_size=3)

    self.conv5 = Conv2d(64, 64, 3, padding=1)
    self.bn5 = BatchNorm2d(64)
    self.conv6 = Conv2d(64, 64, 3, padding=1)
    self.bn6 = BatchNorm2d(64)
    self.maxp3 = MaxPool2d(kernel_size=3)

    self.conv7 = Conv2d(64, 64, 3, padding=1)
    self.bn7 = BatchNorm2d(64)
    self.conv8 = Conv2d(64, 64, 3, padding=1)
    self.bn8 = BatchNorm2d(64)

    self.fc_1 = Linear(576, 64)
    self.fc_last = Linear(64, self.nclasses)

  def forward(self, x):
    x = self.bn1(relu(self.conv1(x)))
    x = self.bn2(relu(self.conv2(x)))
    x = self.maxp(x)
    x = self.bn3(relu(self.conv3(x)))
    x = self.bn4(relu(self.conv4(x)))
    x = self.maxp2(x)
    x = self.bn5(relu(self.conv5(x)))
    x = self.bn6(relu(self.conv6(x)))
    #x = self.maxp3(x)
    x = self.bn7(relu(self.conv7(x)))
    x = self.bn8(relu(self.conv8(x)))
    x = x.view(x.size()[0], -1)
    x = relu(self.fc_1(x))
    x = self.fc_last(x)
    return x

