import torch
import torch.nn as nn
import torch.nn.functional as F

class PasseyNet_mlp(nn.Module):
    def __init__(self, input_size):
        super(PasseyNet_mlp, self).__init__()
        self.fc0 = nn.Linear(6, 236)  # 236X236
        self.fc1 = nn.Linear(236, 256)  # 2X64
        self.fc2 = nn.Linear(256, 6)  # 282X4
        self.fc3 = nn.Linear(6, 1)  # 282X1
        
    def forward(self, x):
        well_log_matrix = F.relu(self.fc0(x))
        well_log_matrix = F.relu(self.fc1(well_log_matrix))
        well_log_matrix = F.relu(self.fc2(well_log_matrix))
        
        RO_matrix = F.relu(self.fc0(x))
        RO_matrix = F.relu(self.fc1(RO_matrix))
        RO_matrix = F.relu(self.fc2(RO_matrix))
        RO_matrix = F.relu(self.fc3(RO_matrix))
        
        Bias_matrix = F.relu(self.fc0(x))
        Bias_matrix = F.relu(self.fc1(Bias_matrix))
        Bias_matrix = F.relu(self.fc2(Bias_matrix))
        Bias_matrix = F.relu(self.fc3(Bias_matrix))
        
        

        output = torch.matmul(torch.matmul(x, well_log_matrix.T),torch.pow(10.0,RO_matrix))+Bias_matrix
        
        return output  