import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGGNet(nn.Module):
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
        super(VGGNet, self).__init__()

        self.model_name = 'VGG'

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.highPassFilter(x)
        x = self.features(x)
        x = x.view(x.size(0), 512 * 4 * 4)
        x = self.classifier(x)
        return x
