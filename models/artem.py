import torch

class Block(torch.nn.Module):

  def __init__(self, in_channels, out_channels, padding):
    super(Block, self).__init__()
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding)
    self.bn = torch.nn.BatchNorm2d(out_channels)
    self.act = torch.nn.ReLU6()
    self.dp = torch.nn.Dropout(p=0.1)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    x = self.dp(x)
    return x

class Block1(torch.nn.Module):

  def __init__(self, in_channels, out_channels):
    super(Block1, self).__init__()
    self.block1 = Block(in_channels, out_channels, 1)
    self.block2 = Block(out_channels, out_channels, 1)
    self.block3 = Block(out_channels, out_channels, 1)

  def forward(self,x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = self.block3(x)
    x = torch.add(x, temp)
    return x

class Block2(torch.nn.Module):

  def __init__(self, in_channels, out_channels):
    super(Block2, self).__init__()
    self.block1 = Block(in_channels, out_channels, 0)
    self.block2 = Block(out_channels, out_channels, 1)
    self.block3 = Block(out_channels, out_channels, 1)

  def forward(self, x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = self.block3(x)
    x = torch.add(x, temp)
    return x
  
########################################
# class with your model
class Net5(torch.nn.Module):
  def __init__(self):
    super(Net5, self).__init__()
    self.block1 = Block1(1,4)
    self.block2 = Block1(4,8)
    self.block3 = Block1(8,16)
    self.pool1 = torch.nn.Sequential(
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU6()
    )
    self.block4 = Block1(16,32)
    self.block5 = Block2(32, 64)
    self.block6 = Block2(64,128)
    self.pool2 = torch.nn.Sequential(
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU6()
    )
    self.fc1 = torch.nn.Linear(128*5*5,4096)
    self.act1 = torch.nn.ReLU6()
    self.fc2 = torch.nn.Linear(4096, 4096)
    self.act2 = torch.nn.ReLU6()
    self.fc3 = torch.nn.Linear(4096,35)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.pool1(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.pool2(x)
    x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.fc3(x)
    return x