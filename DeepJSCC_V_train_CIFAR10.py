
# import numpy as np
import torch
import torch.nn as nn
import os

from utils import Load_cifar10_data, DatasetFolder
from models import ADJSCC_V


BATCH_SIZE = 128
EPOCHS = 400 
LEARNING_RATE = 1e-4
PRINT_RREQ = 150

CHANNEL = 'AWGN'  # Choose AWGN or Fading
IMG_SIZE = [3, 32, 32]
N_channels = 256
Kernel_sz = 5

x_train, x_test = Load_cifar10_data()

train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

current_epoch = 0
CONTINUE_TRAINING = False

enc_out_shape = [48, IMG_SIZE[1]//4, IMG_SIZE[2]//4]
KSZ = str(Kernel_sz)+'x'+str(Kernel_sz)+'_'
if __name__ == '__main__':

    DeepJSCC_V = ADJSCC_V(enc_out_shape, Kernel_sz, N_channels).cuda()
    # DeepJSCC_V = nn.DataParallel(DeepJSCC_V)
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(DeepJSCC_V.parameters(), lr=LEARNING_RATE)    
    
    bestLoss = 1e3  
    if CONTINUE_TRAINING == True:
        DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20.pth.tar')['state_dict'])
        current_epoch = 204
        
    # bestLoss = 1e3   
    for epoch in range(current_epoch, EPOCHS):
        DeepJSCC_V.train()
        print('========================')
        print('lr:%.4e'%optimizer.param_groups[0]['lr'])    
 
        # Model training
        for i, x_input in enumerate(train_loader):
            x_input = x_input.cuda()
            
            SNR_TRAIN = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
            CR = 0.1+0.9*torch.rand(x_input.shape[0], 1).cuda()
            x_rec =  DeepJSCC_V(x_input, SNR_TRAIN, CR, CHANNEL)
            
            loss = criterion(x_input, x_rec) 
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % PRINT_RREQ == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
        

        # Model Evaluation
        DeepJSCC_V.eval()
        totalLoss = 0
        with torch.no_grad():
            for i, test_input in enumerate(test_loader):
                test_input = test_input.cuda()
                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                CR = 0.1+0.9*torch.rand(test_input.shape[0], 1).cuda()
                test_rec =  DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)
            
                totalLoss += criterion(test_input, test_rec).item() * test_input.size(0) 
            averageLoss = totalLoss / (len(test_dataset))
            print('averageLoss=', averageLoss)
            if averageLoss < bestLoss:
                # Model saving
                if not os.path.exists('./JSCC_models'): 
                    os.makedirs('./JSCC_models')
                torch.save({'state_dict': DeepJSCC_V.state_dict(), }, './JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20.pth.tar')
                print('Model saved')
                bestLoss = averageLoss

    print('Training for DeepJSCC_V is finished!')































