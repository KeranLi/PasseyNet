import torch
import torch.nn as nn
import torch.nn.functional as F

class WellLogNet(nn.Module):
    def __init__(self):
        super(WellLogNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 3)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = self.fc1(x)
        return x

class RONet(nn.Module):
    def __init__(self):
        super(RONet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = self.fc1(x)
        return x

class BiasNet(nn.Module):
    def __init__(self):
        super(BiasNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = self.fc1(x)
        return x


class PasseyNet(nn.Module):
    def __init__(self):
        super(PasseyNet, self).__init__()
        self.well_log_net = WellLogNet()
        self.ro_net = RONet()
        self.bias_net = BiasNet()

    def forward(self, x):
        well_log_matrix = self.well_log_net(x)
        ro_matrix = self.ro_net(x)
        bias_matrix = self.bias_net(x)
        
        output = torch.matmul(torch.matmul(x, well_log_matrix.T), 
                              torch.pow(10.0, ro_matrix)) + bias_matrix
        return output