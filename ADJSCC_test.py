
import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils import *
from models import *


BATCH_SIZE = 256
EPOCHS = 150 
LEARNING_RATE = 1e-3
PRINT_RREQ = 250

CHANNEL = 'Fading'  # Choose AWGN or Fading
# if CHANNEL == 'AWGN':
#     # CR_INDEX = torch.Tensor([ 6, 3, 2, 1]).int()
#     # CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()
#     CR_INDEX = [6, 3, 2]
# elif CHANNEL == 'Fading':
#     # CR_INDEX = [3, 3/2]
#     # CR_INDEX = torch.Tensor([3, 3/2]).int()
#     CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()

IMG_SIZE = [3, 32, 32]
N_channels = 256
kernel_sz = 5

enc_shape = [32, 8, 8]
CR = 96//enc_shape[0] # The real compression ration R = 1/CR

_, x_test = Load_cifar100_data()
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

KSZ = '_'+str(kernel_sz)+'x'+str(kernel_sz)+'_'
PSNR_ave = np.zeros((10, 10))
if __name__ == '__main__':
    for m in range(0, 10):
        # enc_shape = [96//CR_INDEX[m], 8, 8]
        DeepJSCC = ADJSCC(enc_shape, kernel_sz, N_channels).cuda()
        # DeepJSCC = nn.DataParallel(DeepJSCC)
        
        DeepJSCC.load_state_dict(torch.load('./JSCC_models/DeepJSCC'+KSZ+CHANNEL+'_'+str(CR)+'_'+str(N_channels)+'.pth.tar')['state_dict'])        
        
        for k in range(0, 10):
            print('Evaluating DeepJSCC with CR = '+str(CR)+' and SNR = '+str(3*k-3)+'dB')
            total_psnr = 0
            DeepJSCC.eval()    
            with torch.no_grad():
                for i, test_input in enumerate(test_loader):
                    SNR = 3*(k-1)*torch.ones((test_input.shape[0], 1))
                    test_input = test_input.cuda()
                    
                    test_rec = DeepJSCC(test_input, SNR, CHANNEL)
                    
                    test_input = Img_transform(test_input)
                    test_rec  = Img_transform(test_rec)
                    psnr_ave = Compute_batch_PSNR(test_input, test_rec)
                    total_psnr += psnr_ave      
                averagePSNR = total_psnr / i  
                print('PSNR = ' + str(averagePSNR)) 
                
            PSNR_ave[m, k] = averagePSNR






