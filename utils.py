
import numpy as np
# import torch
# from torchvision import datasets, transforms
from torch.utils.data import Dataset

# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_pnsr

# from models import *


# Note that the original data is downloaded from keras.datasets, not from torch.utils.data
def Load_cifar10_data():
    x_train = np.load('data/CIFAR10_raw/x_train.npy')
    x_test = np.load('data/CIFAR10_raw/x_test.npy')
    # from keras.datasets import cifar10
    # (x_train, y_train_), (x_test, y_test_) = cifar10.load_data()        
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test


# Note that the original data is downloaded from keras.datasets, not from torch.utils.data
def Load_cifar100_data():
    x_train = np.load('data/CIFAR100_raw/x_train.npy')
    x_test = np.load('data/CIFAR100_raw/x_test.npy')
    # from keras.datasets import cifar10
    # (x_train, y_train_), (x_test, y_test_) = cifar10.load_data()        
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test


# def Plot_CIFAR10_img(x):
#     digit_size = 32
#     n = 5
#     figure = np.zeros((digit_size*n, digit_size * n, 3))
#     for i in range (n):
#         x_i = x[i * n: (i + 1) * n, :]
#         for j in range(n):
#             digit = x_i[j].reshape(digit_size, digit_size, 3)
#             figure[i * digit_size: (i + 1) * digit_size,
#                     j * digit_size: (j + 1) * digit_size, :] = digit

#     plt.figure(figsize=(10, 10))
#     plt.imshow(figure, cmap='Greys_r')
#     plt.axis('off')
#     plt.show()  
    

def Img_transform(test_rec):
    test_rec = test_rec.permute(0, 2, 3, 1)
    test_rec = test_rec.cpu().detach().numpy()
    test_rec = test_rec*255
    test_rec = test_rec.astype(np.uint8)
    return test_rec

def Compute_batch_PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0]))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    psnr_ave = np.mean(psnr_i1)
    return psnr_ave


def Compute_IMG_PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0], 1))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    return psnr_i1

# Data Loader  
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]

# Use the following learning schedulars maybe helpful for improving the training quality
def lr_schedular(cur_epoch, warmup_epoch, epochs, lr_max):
    lr_min = 1e-6 
    kappa = (lr_max-lr_min)/warmup_epoch
    if cur_epoch < warmup_epoch:
        lr = lr_min + kappa*cur_epoch
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (cur_epoch-warmup_epoch) / epochs))
    return lr

def lr_schedular_step(epoch, warmup_epoch, EPOCHS, lr_max):
    lr_min = 1e-6 
    kappa = (lr_max-lr_min)/warmup_epoch
    if epoch < warmup_epoch:
        lr = lr_min + kappa*epoch
    else:    
        eta = EPOCHS/100
        if epoch<=25*eta:
            lr = lr_max
        elif epoch>25*eta and epoch<=50*eta:
            lr = lr_max/2
        elif epoch>50*eta and epoch<=80*eta:
            lr = lr_max/4
        elif epoch>80*eta and epoch<=95*eta:
            lr = lr_max/8
        elif epoch>95*eta: 
            lr = lr_max/16
    return lr



















