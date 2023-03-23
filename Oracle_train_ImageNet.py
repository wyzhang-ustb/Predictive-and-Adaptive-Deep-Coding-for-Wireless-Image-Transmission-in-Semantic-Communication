
import torch
import torch.nn as nn
import os

from utils import *
from models import *
from OracleNet import OracleNet


from data_loader import train_data_loader, test_data_loader

BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 1e-4
PRINT_RREQ = 150


IMGZ = 128
train_loader = train_data_loader(batch_size = BATCH_SIZE, imgz = IMGZ, workers = 2)
test_loader = test_data_loader(batch_size = BATCH_SIZE, imgz = IMGZ, workers = 2)


CHANNEL = 'AWGN'  # Choose AWGN or Fading
IMG_SIZE = [3, IMGZ, IMGZ]
N_channels = 256
kernel_sz = 5
enc_shape = [48, IMGZ//4, IMGZ//4]
KSZ = str(kernel_sz)+'x'+str(kernel_sz)+'_'

DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
# DeepJSCC_V = nn.DataParallel(DeepJSCC_V)
DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_ImageNet.pth.tar')['state_dict'])
DeepJSCC_V.eval()  

OraNet = OracleNet(enc_shape[0]).cuda()
# OraNet = nn.DataParallel(OraNet)
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(OraNet.parameters(), lr=LEARNING_RATE)  

if __name__ == '__main__':
    bestLoss = 1e3
    for epoch in range(EPOCHS):
        OraNet.train()
        for i, (x_input, _) in enumerate(train_loader):
            SNR = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
            CR = 0.1+0.9*torch.rand(x_input.shape[0], 1).cuda()
            
            x_input = torch.Tensor(x_input).cuda()        
            x_rec = DeepJSCC_V(x_input, SNR, CR, CHANNEL)
            # z = DeepJSCC_V.module.encoder(x_input, SNR)
            
            z = DeepJSCC_V.encoder(x_input, SNR)
            
            x_input = Img_transform(x_input)
            x_rec  = Img_transform(x_rec)
            psnr_batch = Compute_IMG_PSNR(x_input, x_rec)
            psnr_batch = torch.Tensor(psnr_batch).cuda()

            z = z.view(-1, enc_shape[0], IMGZ//4, IMGZ//4)
            psnr_pred = OraNet(z, SNR, CR)
            
            loss = criterion(psnr_batch, psnr_pred) 
            loss = loss.mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % PRINT_RREQ == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))


        # Model Evaluation
        OraNet.eval()
        totalLoss = 0
        with torch.no_grad():
            for i, (test_input, _) in enumerate(test_loader):
                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                CR = 0.1+0.9*torch.rand(test_input.shape[0], 1).cuda()
                
                test_input = torch.Tensor(test_input).cuda()
                test_rec =  DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)
                # z = DeepJSCC_V.module.encoder(test_input, SNR_TEST)
                z = DeepJSCC_V.encoder(test_input, SNR_TEST)
                
                test_input = Img_transform(test_input)
                test_rec  = Img_transform(test_rec)
                psnr_batch = Compute_IMG_PSNR(test_input, test_rec)
                psnr_batch = torch.Tensor(psnr_batch).cuda()   
                
                z = z.view(-1, enc_shape[0], IMGZ//4, IMGZ//4)
                psnr_pred = OraNet(z, SNR_TEST, CR)
                
                totalLoss += criterion(psnr_batch, psnr_pred).item() * psnr_batch.size(0) 
            averageLoss = totalLoss / 5000
            print('averageLoss=', averageLoss)
            if averageLoss < bestLoss:
                # Model saving
                if not os.path.exists('./JSCC_models'): 
                    os.makedirs('./JSCC_models')
                torch.save({'state_dict': OraNet.state_dict(), }, './JSCC_models/OracleNet_'+CHANNEL+'_ImageNet.pth.tar')
                print('Model saved')
                bestLoss = averageLoss














