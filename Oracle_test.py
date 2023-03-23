
import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils import *
from models import *
from OracleNet import OracleNet


BATCH_SIZE = 128
EPOCHS = 150 
LEARNING_RATE = 1e-4
PRINT_RREQ = 150


_, x_test = Load_cifar100_data()
# train_dataset = DatasetFolder(x_train)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


CHANNEL = 'Fading'  # Choose AWGN or Fading
CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()

IMG_SIZE = [3, 32, 32]
N_channels = 256
kernel_sz = 5
enc_shape = [48, 8, 8]
KSZ = str(kernel_sz)+'x'+str(kernel_sz)+'_'

DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
# DeepJSCC_V = nn.DataParallel(DeepJSCC_V)

DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_cifar10.pth.tar')['state_dict'])
DeepJSCC_V.eval()  

OraNet = OracleNet(enc_shape[0]).cuda()
OraNet.load_state_dict(torch.load('./JSCC_models/OracleNet_'+CHANNEL+'_Res.pth.tar')['state_dict'])
OraNet.eval()

criterion = nn.MSELoss().cuda()
MSE_pred = np.zeros((10, 10))
if __name__ == '__main__':
    # Model Evaluation
    for m in range(0, 10):
        cr = 1/CR_INDEX[m]    
        for k in range(0, 10):
            totalLoss = 0
            with torch.no_grad():
                for i, test_input in enumerate(test_loader):
                    SNR_TEST = 3*(k-1)*torch.ones((test_input.shape[0], 1)).cuda()
                    CR = cr*torch.ones((test_input.shape[0], 1)).cuda()

                    test_input = torch.Tensor(test_input).cuda()
                    test_rec =  DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)
                    z = DeepJSCC_V.module.encoder(test_input, SNR_TEST)
                    
                    test_input = Img_transform(test_input)
                    test_rec  = Img_transform(test_rec)
                    psnr_batch = Compute_IMG_PSNR(test_input, test_rec)
                    psnr_batch = torch.Tensor(psnr_batch).cuda()   
                    
                    z = z.view(-1, enc_shape[0], 8, 8)
                    psnr_pred = OraNet(z, SNR_TEST, CR)
                    
                    totalLoss += criterion(psnr_batch, psnr_pred).item() * psnr_batch.size(0) 
                averageLoss = totalLoss / (len(test_dataset))
                print('CR = '+str(cr.item())+ ', SNR = '+ str(3*(k-1)) +', MSE =', averageLoss)
            
            MSE_pred[m, k] = averageLoss


# a = psnr_batch.cpu().numpy()
# b = psnr_pred.cpu().numpy()










