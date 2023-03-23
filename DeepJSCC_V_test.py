
import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils import *
from models import *

from data_loader import train_data_loader, test_data_loader

BATCH_SIZE = 256
EPOCHS = 150 
LEARNING_RATE = 1e-3
PRINT_RREQ = 250

CHANNEL = 'Fading'  # Choose AWGN or Fading
if CHANNEL == 'AWGN':
    # CR_INDEX = torch.Tensor([ 6, 3, 2, 1]).int()
    # CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()
    CR_INDEX = torch.Tensor([6, 3, 2]).int()
elif CHANNEL == 'Fading':
    # CR_INDEX = [3, 3/2]
    # CR_INDEX = torch.Tensor([3, 3/2]).int()
    CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()
    # CR_INDEX = torch.Tensor([2, 1]).int()

IMG_SIZE = [32, 32, 32]
N_channels = 256
kernel_sz = 5
KSZ = str(kernel_sz)+'x'+str(kernel_sz)+'_'


_, x_test = Load_cifar100_data()
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


IMGZ = 32
c_max = 48
enc_shape = [c_max, IMGZ//4, IMGZ//4]

PSNR_ave = np.zeros((10, 10))
if __name__ == '__main__':
    for m in range(0, 10):
        # cr = 1/CR_INDEX[m]
        cr = 2/3
        DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
        # DeepJSCC_V = nn.DataParallel(DeepJSCC_V)
        
        DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_cifar10.pth.tar')['state_dict'])
        
        for k in range(0, 10):
            print('Evaluating DeepJSCC_VLC with CR = '+str(2*CR_INDEX[m].item())+' and SNR = '+str(3*k-3)+'dB')
            total_psnr = 0
            DeepJSCC_V.eval()    
            with torch.no_grad():
                # for i, (test_input,_) in enumerate(test_loader):
                for i, test_input in enumerate(test_loader):    
                    SNR = 3*(k-1)*torch.ones((test_input.shape[0], 1))
                    CR = cr*torch.ones((test_input.shape[0], 1))
                    SNR = SNR.cuda()
                    CR = CR.cuda()
                    test_input = test_input.cuda()
                    
                    test_rec = DeepJSCC_V(test_input, SNR, CR, CHANNEL)
                    
                    test_input = Img_transform(test_input)
                    test_rec  = Img_transform(test_rec)
                    psnr_ave = Compute_batch_PSNR(test_input, test_rec)
                    total_psnr += psnr_ave      
                averagePSNR = total_psnr / i  
                print('PSNR = ' + str(averagePSNR)) 
                
            PSNR_ave[m, k] = averagePSNR






