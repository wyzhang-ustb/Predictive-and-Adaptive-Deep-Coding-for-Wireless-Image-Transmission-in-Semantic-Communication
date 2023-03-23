

import matplotlib.pyplot as plt # plt 用于显示图片
# import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

import torch
from utils import *
from models import *
from OracleNet import OracleNet

from skimage.metrics import peak_signal_noise_ratio as compute_pnsr



CHANNEL = 'AWGN'
Rep_N = 1
	
# CR_LEVELS = [20,18,16,14,12,10,8,6,4,3]
CR_LEVELS = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3]

# select the target Kodak image
k = 13


if k<10:
    img_id = '0'+str(k)
else:
    img_id = str(k)    
img_file = './data/Kodak24/kodim'+img_id+'.png'
img = Image.open(img_file) 

plt.imshow(img)
plt.axis('off')
plt.show()

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

PSNR_ALL = np.zeros((18, 9))
PSNR_PRED_ALL = np.zeros((18, 9))

N_batch = 32
for cr_index in range(0, 18):    
    cr0 = 2/CR_LEVELS[cr_index]
    cr = cr0*torch.ones((N_batch, 1)).cuda() 
    
    for snr_index in range(0, 9):
        snr = snr_index*3*torch.ones((N_batch, 1)).cuda()
        
        PSNR_REP = np.zeros((Rep_N, 1))
        with torch.no_grad():
            for i in range(0, Rep_N):
                # print(i)                         
                img_i = img.tile(N_batch, 1, 1, 1)
                img_rec = DeepJSCC_V(img_i, snr, cr, CHANNEL)
        
                img0 = Img_transform(img_i)
                img_rec  = Img_transform(img_rec)
                PSNR = Compute_batch_PSNR(img0, img_rec)
                
                PSNR_REP[i, 0] = PSNR
        
        snr1 = torch.Tensor([snr_index*3, ]).cuda()
        cr1 = torch.Tensor([cr0, ]).cuda() 
        # z = DeepJSCC_V.module.encoder(img, snr1)
        z = DeepJSCC_V.encoder(img, snr1)
        
        z = z.view(-1, enc_shape[0], input_shape[1]//4, input_shape[2]//4)
        PSNR_pred = OraNet(z, snr1, cr1)
        PSNR_pred = PSNR_pred[0].cpu().detach().numpy()
        
        PSNR_ALL[cr_index, snr_index] = np.mean(PSNR)
        PSNR_PRED_ALL[cr_index, snr_index] = PSNR_pred
        
        print(np.mean(PSNR))
        print(PSNR_pred)



# ave1 = np.mean(PSNR_Kodak, 0)
# ave2 = np.mean(PSNR_Kodak_pred, 0)


# plt.imshow(img)
# plt.axis('off')
# plt.show()

# plt.imshow(img_rec[0, :])
# plt.axis('off')
# plt.show()


