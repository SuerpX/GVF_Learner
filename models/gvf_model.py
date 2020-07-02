import torch
import torch.nn as nn

class gvf_model(nn.Module):
    def __init__(self, input_len, output_len):
        super(gvf_model, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, output_len)
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x