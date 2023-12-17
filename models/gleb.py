import torch

class LeNet(torch.nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv_add1 = torch.nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = 3, padding = 1)
    self.batn_add1 = torch.nn.BatchNorm2d(2)
    self.act_add1 = torch.nn.ReLU6()
    self.drop_add1 = torch.nn.Dropout(p = 0.1)

    self.conv_add2 = torch.nn.Conv2d(in_channels = 2, out_channels = 2, kernel_size = 3, padding = 1)
    self.batn_add2 = torch.nn.BatchNorm2d(2)
    self.act_add2 = torch.nn.ReLU6()
    self.drop_add2 = torch.nn.Dropout(p = 0.1)

    self.conv_add3 = torch.nn.Conv2d(in_channels = 2, out_channels = 2, kernel_size = 3, padding = 1)
    self.batn_add3 = torch.nn.BatchNorm2d(2)
    self.act_add3 = torch.nn.ReLU6()
    self.drop_add3 = torch.nn.Dropout(p = 0.1)

    self.conv_add4 = torch.nn.Conv2d(in_channels = 2, out_channels = 4, kernel_size = 3, padding = 1)
    self.batn_add4 = torch.nn.BatchNorm2d(4)
    self.act_add4 = torch.nn.ReLU6()
    self.drop_add4 = torch.nn.Dropout(p = 0.1)

    self.conv_add5 = torch.nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, padding = 1)
    self.batn_add5 = torch.nn.BatchNorm2d(4)
    self.act_add5 = torch.nn.ReLU6()
    self.drop_add5 = torch.nn.Dropout(p = 0.1)

    self.conv_add6 = torch.nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, padding = 1)
    self.batn_add6 = torch.nn.BatchNorm2d(4)
    self.act_add6 = torch.nn.ReLU6()
    self.drop_add6 = torch.nn.Dropout(p = 0.1)




    self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, padding = 1)
    self.batn1 = torch.nn.BatchNorm2d(4)
    self.act1 = torch.nn.ReLU6()
    self.drop1 = torch.nn.Dropout(p = 0.1)

    self.conv2 = torch.nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, padding = 0)
    self.batn2 = torch.nn.BatchNorm2d(8)
    self.act2 = torch.nn.ReLU6()
    self.drop2 = torch.nn.Dropout(p = 0.1)

    self.conv3 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
    self.batn3 = torch.nn.BatchNorm2d(8)
    self.act3 = torch.nn.ReLU6()
    self.drop3 = torch.nn.Dropout(p = 0.1)

    self.conv4 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 0)
    self.batn4 = torch.nn.BatchNorm2d(16)
    self.act4 = torch.nn.ReLU6()
    self.drop4 = torch.nn.Dropout(p = 0.1)

    self.conv5 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
    self.batn5 = torch.nn.BatchNorm2d(16)
    self.act5 = torch.nn.ReLU6()
    self.drop5 = torch.nn.Dropout(p = 0.1)

    self.pool1 = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)
    self.act6 = torch.nn.ReLU6()

    self.conv6 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
    self.batn6 = torch.nn.BatchNorm2d(32)
    self.act7 = torch.nn.ReLU6()
    self.drop6 = torch.nn.Dropout(p = 0.1)

    self.conv7 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 0)
    self.batn7 = torch.nn.BatchNorm2d(32)
    self.act8 = torch.nn.ReLU6()
    self.drop7 = torch.nn.Dropout(p = 0.1)

    self.conv8 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self.batn8 = torch.nn.BatchNorm2d(64)
    self.act9 = torch.nn.ReLU6()
    self.drop8 = torch.nn.Dropout(p = 0.1)

    self.conv9 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
    self.batn9 = torch.nn.BatchNorm2d(64)
    self.act10 = torch.nn.ReLU6()
    self.drop9 = torch.nn.Dropout(p = 0.1)

    self.pool2 = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)
    self.act11 = torch.nn.ReLU6()
    self.fc1 = torch.nn.Linear(1600, 800)
    self.act12 = torch.nn.ReLU6()
    self.fc2 = torch.nn.Linear(800, 400)
    self.act13 = torch.nn.ReLU6()
    self.fc3 = torch.nn.Linear(400, 35)

  def forward(self, x):
    x = self.conv_add1(x)
    x = self.batn_add1(x)
    x = self.act_add1(x)
    x = self.drop_add1(x)

    skip_conection1 = x
    x = self.conv_add2(x)
    x = self.batn_add2(x)
    x = self.act_add2(x)
    x = self.drop_add2(x)

    x = self.conv_add3(x)
    x = self.batn_add3(x)
    x = self.act_add3(x)
    x = self.drop_add3(x)
    x = torch.add(x, skip_conection1)

    x = self.conv_add4(x)
    x = self.batn_add4(x)
    x = self.act_add4(x)
    x = self.drop_add4(x)

    skip_conection2 = x
    x = self.conv_add5(x)
    x = self.batn_add5(x)
    x = self.act_add5(x)
    x = self.drop_add5(x)

    x = self.conv_add6(x)
    x = self.batn_add6(x)
    x = self.act_add6(x)
    x = self.drop_add6(x)
    x = torch.add(x, skip_conection2)




    x = self.conv1(x)
    x = self.batn1(x)
    x = self.act1(x)
    x = self.drop1(x)

    x = self.conv2(x)
    x = self.batn2(x)
    x = self.act2(x)
    x = self.drop2(x)

    x = self.conv3(x)
    x = self.batn3(x)
    x = self.act3(x)
    x = self.drop3(x)

    x = self.conv4(x)
    x = self.batn4(x)
    x = self.act4(x)
    x = self.drop4(x)

    x = self.conv5(x)
    x = self.batn5(x)
    x = self.act5(x)
    x = self.drop5(x)

    x = self.pool1(x)
    x = self.act6(x)

    x = self.conv6(x)
    x = self.batn6(x)
    x = self.act7(x)
    x = self.drop6(x)

    x = self.conv7(x)
    x = self.batn7(x)
    x = self.act8(x)
    x = self.drop7(x)

    x = self.conv8(x)
    x = self.batn8(x)
    x = self.act9(x)
    x = self.drop8(x)

    x = self.conv9(x)
    x = self.batn9(x)
    x = self.act10(x)
    x = self.drop9(x)

    x = self.pool2(x)
    x = self.act11(x)

    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    x = self.fc1(x)
    x = self.act12(x)
    x = self.fc2(x)
    x = self.act13(x)
    x = self.fc3(x)

    return x