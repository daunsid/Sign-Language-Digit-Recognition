import data

import torch
import torch.nn as nn
import torch.nn.functional as F



class SignNet(nn.Module):
    def __init__(self, in_channels=data.input_layer()[1], out_channels=6):
        super(SignNet, self).__init__()
        
        self.Conv1 = nn.Conv2d(in_channels, out_channels, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.Conv2 = nn.Conv2d(out_channels, 16, 5)
        self.Conv3 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46*12*12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = self.pool(F.relu(self.Conv2(x)))
        x = self.pool(F.relu(self.Conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def softmax(x): return torch.exp(x)/torch.exp(x).sum(dim=1,keepdim=True)

def criterion(prediction, true_value):
    sm_pred = softmax(prediction)
    idx = range(len(prediction))
    return -torch.log(sm_pred[idx,true_value]).mean()

def get_lr(): return 0.01
def get_epoch(): return 20

def init_model_params():
    model = SignNet()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=get_lr())
    return model, optimizer

