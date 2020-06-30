import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FakeNet(nn.Module):

    def highPassFilter(self, x):
        HPF = torch.Tensor([[0,0,0],[-1,1,0],[0,0,0]])
        HPF = HPF.reshape(1,1,HPF.shape[0],HPF.shape[1])
        HPF = HPF.to(device)
        outputs = torch.zeros(x.size()).to(device)
        for i in range(3):
            ChanFilt = F.conv2d(x[:,i,:,:].unsqueeze(1),HPF, padding=1) 
            outputs[:,i,:,:] = ChanFilt.squeeze(1)

        return outputs

    def __init__(self):
        super(FakeNet, self).__init__()
        self.model_name = 'FakeNet'

        self.group1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),    
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.highPassFilter(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = x.view(x.size(0), 128 * 16 * 16)
        x = self.classifier(x)

        return x

