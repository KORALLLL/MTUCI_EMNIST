import torch

class OneBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, padding):
    super(OneBlock, self).__init__()
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
  
class BlockSimilarSize(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BlockSimilarSize, self).__init__()
    self.block1 = OneBlock(in_channels, out_channels, 1)
    self.block2 = OneBlock(out_channels, out_channels, 1)

  def forward(self,x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = torch.add(x, temp)
    return x
  
class BlockVariousSize(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BlockVariousSize, self).__init__()
    self.block1 = OneBlock(in_channels, out_channels, 0)
    self.block2 = OneBlock(out_channels, out_channels, 1)

  def forward(self,x):
    x = self.block1(x)
    temp = x
    x = self.block2(x)
    x = torch.add(x, temp)
    return x
  
class Lenet4(torch.nn.Module):
  def __init__(self):
    super(Lenet4, self).__init__()

    self.block1 =  BlockSimilarSize(1,4)


    self.block3 =  BlockSimilarSize(4,8)
    self.block4 =  BlockSimilarSize(8,16)



    self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.act5 = torch.nn.ReLU6()

    self.block5 = BlockVariousSize(16, 32)


    self.block7 = BlockVariousSize(32, 64)
    self.block8 =  BlockSimilarSize(64,128)




    self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.act10 = torch.nn.ReLU6()

    self.fc1 = torch.nn.Linear(128*5*5, 1000)
    self.act11 = torch.nn.ReLU6()

    self.fc2 = torch.nn.Linear(1000, 1000)
    self.act12 = torch.nn.ReLU6()

    self.fc3 = torch.nn.Linear(1000, 35)

  def forward(self, x):
    x = self.block1(x)

    x = self.block3(x)
    x = self.block4(x)

    x = self.pool1(x)
    x = self.act5(x)

    x = self.block5(x)

    x = self.block7(x)
    x = self.block8(x)

    x = self.pool2(x)
    x = self.act10(x)
    x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
    x = self.fc1(x)
    x = self.act11(x)
    x = self.fc2(x)
    x = self.act12(x)
    x = self.fc3(x)
    return x
