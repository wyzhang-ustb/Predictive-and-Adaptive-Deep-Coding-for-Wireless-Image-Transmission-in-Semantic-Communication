 

import matplotlib.pyplot as plt 
# import matplotlib.image as mpimg 
import numpy as np
from PIL import Image

import torch
from utils import *
from models import *
from OracleNet import OracleNet

from skimage.metrics import peak_signal_noise_ratio as compute_pnsr

CR = 1/10

CHANNEL = 'AWGN' # Choose AWGN or Fading
Rep_N = 1


# def kodak_test(CR):
PSNR_Kodak = np.zeros((24, 9))
PSNR_Kodak_pred = np.zeros((24, 9))
for k in range(0, 24):
    print('Image ' + str(k))
    if k<9:
        img_id = '0'+str(k+1)
    else:
        img_id = str(k+1)    
    img_file = './data/Kodak24/kodim'+img_id+'.png'
    img = Image.open(img_file) 
    
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255
    
    img = torch.Tensor(img).cuda()
    img = img.unsqueeze(0)
    
    
    input_shape = np.shape(img)[1:4]
    kernel_sz = 5
    N_channels = 256
    KSZ = str(kernel_sz)+'x'+str(kernel_sz)+'_'
    enc_shape = [48, input_shape[1]//4, input_shape[2]//4]
    
    DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
    # DeepJSCC_V = nn.DataParallel(DeepJSCC_V)
    
    DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_ImageNet.pth.tar')['state_dict'])
    DeepJSCC_V.eval()  
    
    OraNet = OracleNet(enc_shape[0]).cuda()
    OraNet.load_state_dict(torch.load('./JSCC_models/OracleNet_'+CHANNEL+'_ImageNet.pth.tar')['state_dict'])
    OraNet.eval()
    
    for snr_index in range(0, 9):
        PSNR_REP = np.zeros((Rep_N, 1))
        with torch.no_grad():
            for i in range(0, Rep_N):
                # print(i)
                # snr = torch.Tensor([snr_index*3, ]).cuda()
                # cr = torch.Tensor([CR, ]).cuda()
                
                snr = snr_index*3*torch.ones((5, 1)).cuda()
                cr = CR*torch.ones((5, 1)).cuda()               
                
                img_i = img.tile(5, 1, 1, 1)
                img_rec = DeepJSCC_V(img_i, snr, cr, CHANNEL)
        
                img0 = Img_transform(img_i)
                img_rec  = Img_transform(img_rec)
                PSNR = Compute_batch_PSNR(img0, img_rec)
                
                PSNR_REP[i, 0] = PSNR
        
        snr1 = torch.Tensor([snr_index*3, ]).cuda()
        cr1 = torch.Tensor([CR, ]).cuda() 
        # z = DeepJSCC_V.module.encoder(img, snr1)
        z = DeepJSCC_V.encoder(img, snr1)
        z = z.view(-1, enc_shape[0], input_shape[1]//4, input_shape[2]//4)
        PSNR_pred = OraNet(z, snr1, cr1)
        PSNR_pred = PSNR_pred[0].cpu().detach().numpy()
        
        PSNR_Kodak[k, snr_index] = np.mean(PSNR)
        PSNR_Kodak_pred[k, snr_index] = PSNR_pred
        
        print(np.mean(PSNR))
        print(PSNR_pred)

ave1 = np.mean(PSNR_Kodak, 0)
ave2 = np.mean(PSNR_Kodak_pred, 0)


# return ave1, ave2


# CR_all = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 3/2]).int()

# # CR = 1/8

# CHANNEL = 'Fading'
# Rep_N = 2

# psnr_all = np.zeros((10, 9))
# psnr_pred_all = np.zeros((10, 9))
# for i in range(0, 10):
#     CR = 1/CR_all[i]
#     ave1, ave2 = kodak_test(CR)
#     psnr_all[i,:] = ave1
#     psnr_pred_all[i,:] = ave2
    
#     print('Evaluation ' + str(CR) + 'finished...')



# plt.imshow(img)
# plt.axis('off')
# plt.show()

# plt.imshow(img_rec[0, :])
# plt.axis('off')
# plt.show()


