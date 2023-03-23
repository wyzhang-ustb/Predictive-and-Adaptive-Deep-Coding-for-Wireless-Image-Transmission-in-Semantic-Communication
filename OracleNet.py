
import numpy as np
import torch.nn as nn
import torch



class fc_ResBlock(nn.Module):
    def __init__(self, Nin, Nout):
        super(fc_ResBlock, self).__init__()
        Nh = Nin*2
        self.use_fc3 = False
        self.fc1 = nn.Linear(Nin, Nh)
        self.fc2 = nn.Linear(Nh, Nout)
        self.relu = nn.ReLU()
        if Nin != Nout:
            self.use_fc3 = True
            self.fc3 = nn.Linear(Nin, Nout)
    def forward(self, x): 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.use_fc3 == True:
            x = self.fc3(x)
        out = out+x
        out = self.relu(out)
        return out 

# The oracle network for predicting the PSNR of the transmitted images
class OracleNet(nn.Module):
    def __init__(self, Nc_max):
        super(OracleNet, self).__init__()
        self.fc1 = fc_ResBlock(Nc_max*2+2, Nc_max)
        self.fc2 = fc_ResBlock(Nc_max, Nc_max)
        self.fc3 = nn.Linear(Nc_max, 1)
        self.relu = nn.ReLU()
    def forward(self, x, snr, cr):
        N_out = torch.round(48*cr).int()
        if snr.shape[0]==1:
            snr = snr.unsqueeze(1)  
            N_out = N_out.unsqueeze(1)
        std_feat = torch.std(x, (2, 3)) 
        mean_feat = torch.mean(x, (2, 3))
        out = torch.cat((mean_feat, std_feat, snr, N_out), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out




