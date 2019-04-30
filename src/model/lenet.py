import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class LeNet5_r(nn.Module):
   def __init__(self, n_branch):
       super(LeNet5_r, self).__init__()
       self.n = n_branch
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
       self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
       self.fc1 = nn.Linear(32*6*6, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, self.n)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
       x = x.view(-1, 32*6*6)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x