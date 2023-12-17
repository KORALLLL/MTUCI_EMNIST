import torch

class OneBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, padding):
    super(OneBlock, self).__init__()
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding)
    self.bn = torch.nn.BatchNorm2d(out_channels)
    self.act = torch.nn.ReLU6()
    self.dp = torch.nn.Dropout(p=0.05)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    x = self.dp(x)
    return x

class BlockSimilarSize(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BlockSimilarSize, self).__init__()
    self.block1 = OneBlock(in_channels, out_channels, 1)
    self.block2 = OneBlock(out_channels, out_channels, 1)
    self.block3 = OneBlock(out_channels, out_channels, 1)
  def forward(self,x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = self.block3(x)
    x = torch.add(x, temp)
    return x

class BlockVariousSize(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BlockVariousSize, self).__init__()
    self.block1 = OneBlock(in_channels, out_channels, 0)
    self.block2 = OneBlock(out_channels, out_channels, 1)
    self.block3 = OneBlock(out_channels, out_channels, 1)
  def forward(self, x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = self.block3(x)
    x = torch.add(x, temp)
    return x


class Model5(torch.nn.Module):
  def __init__(self):
    super(Model5, self).__init__()
    self.block1 = BlockSimilarSize(1,4)
    self.block2 = BlockSimilarSize(4,8)
    self.block3 = BlockSimilarSize(8,16)
    self.pool1 = torch.nn.Sequential(
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU6()
    )
    self.block4 = BlockSimilarSize(16,32)
    self.block5 = BlockVariousSize(32, 64)
    self.block6 = BlockVariousSize(64,128)
    self.pool2 = torch.nn.Sequential(
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU6()
    )
    self.fc1 = torch.nn.Linear(128*5*5,2000)
    self.act1 = torch.nn.ReLU6()
    self.fc2 = torch.nn.Linear(2000, 2000)
    self.act2 = torch.nn.ReLU6()
    self.fc3 = torch.nn.Linear(2000,35)

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