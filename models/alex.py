import torch

class LeNetDropOut(torch.nn.Module):
  def __init__(self, dropout_rate):
    super(LeNetDropOut, self).__init__()

    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=2,
                                 kernel_size=3, padding=1)
    self.act1 = torch.nn.ReLU6()
    self.dropout1 = torch.nn.Dropout(p=dropout_rate)
    self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=4,
                                 kernel_size=3, padding=1)

    self.act2 = torch.nn.ReLU6()
    self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=8,
                                 kernel_size=3, padding=1)
    self.act3 = torch.nn.ReLU6()
    self.dropout3 = torch.nn.Dropout(p=dropout_rate)
    self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=16,
                                 kernel_size=3, padding=1)
    self.act4 = torch.nn.ReLU6()
    self.dropout4 = torch.nn.Dropout(p=dropout_rate)
    self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                 kernel_size=3, padding=1)
    self.act5 = torch.nn.ReLU6()
    self.dropout5 = torch.nn.Dropout(p=dropout_rate)
    self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.act6 = torch.nn.ReLU6()


    self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=32,
                                 kernel_size=3, padding=1)
    self.act7 = torch.nn.ReLU6()
    self.dropout6 = torch.nn.Dropout(p=dropout_rate)
    self.conv7 = torch.nn.Conv2d(in_channels=32, out_channels=32,
                                 kernel_size=3, padding=1)
    self.act8 = torch.nn.ReLU6()
    self.dropout7 = torch.nn.Dropout(p=dropout_rate)
    self.conv8 = torch.nn.Conv2d(in_channels=32, out_channels=32,
                                 kernel_size=3, padding=1)
    self.act9 = torch.nn.ReLU6()
    self.dropout8 = torch.nn.Dropout(p=dropout_rate)
    self.conv9 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                 kernel_size=3, padding=0)
    self.act10 = torch.nn.ReLU6()
    self.dropout9 = torch.nn.Dropout(p=dropout_rate)
    self.conv10 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=3, padding=0)
    self.act11 = torch.nn.ReLU6()
    self.dropout10 = torch.nn.Dropout(p=dropout_rate)
    self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    self.act12 = torch.nn.ReLU6()

    self.fc1 = torch.nn.Linear(5 * 5 * 64, 1000)
    self.act13 = torch.nn.ReLU6()
    self.fc2 = torch.nn.Linear(1000, 1000)
    self.act14 = torch.nn.ReLU6()
    self.fc3 = torch.nn.Linear(1000, 35)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.act2(x)
    x = self.conv3(x)
    x = self.act3(x)
    x = self.conv4(x)
    x = self.act4(x)

    x = self.conv5(x)
    x = self.act5(x)

    x = self.pool1(x)
    x = self.act6(x)
    skip_connection_1 = x
    x = self.conv6(x)
    x = self.act7(x)
    x = self.conv7(x)
    x = self.act8(x)
    x = self.conv8(x)
    x = self.act9(x)
    x = torch.add(x, skip_connection_1)
    x = self.conv9(x)
    x = self.act10(x)
    x = self.conv10(x)
    x = self.act11(x)
    x = self.pool2(x)
    x = self.act12(x)

    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

    x = self.fc1(x)
    x = self.act13(x)
    x = self.fc2(x)
    x = self.act14(x)
    x = self.fc3(x)

    return x
