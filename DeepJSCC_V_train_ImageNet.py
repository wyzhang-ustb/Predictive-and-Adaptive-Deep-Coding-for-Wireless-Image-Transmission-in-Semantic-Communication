

import torch
import torch.nn as nn
import os

from models import ADJSCC_V

from data_loader import train_data_loader, test_data_loader


BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 1e-4
PRINT_RREQ = 50
SAVE_RREQ = 500

CHANNEL = 'AWGN'  # Choose AWGN or Fading
N_channels = 256
Kernel_sz = 5

IMGZ = 128
train_loader = train_data_loader(batch_size = BATCH_SIZE, imgz = IMGZ, workers = 2)
test_loader = test_data_loader(batch_size = BATCH_SIZE, imgz = IMGZ, workers = 2)

current_epoch = 0
CONTINUE_TRAINING = True
LOAD_PRETRAIN = False

enc_out_shape = [48, IMGZ//4, IMGZ//4]
KSZ = str(Kernel_sz)+'x'+str(Kernel_sz)+'_'
if __name__ == '__main__':

    DeepJSCC_V = ADJSCC_V(enc_out_shape, Kernel_sz, N_channels).cuda()
    # DeepJSCC_V = nn.DataParallel(DeepJSCC_V)
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(DeepJSCC_V.parameters(), lr=LEARNING_RATE)    

    if LOAD_PRETRAIN == True:
        DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_cifar10.pth.tar')['state_dict'])  

    bestLoss = 1e3  
    if CONTINUE_TRAINING == True:
        DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_ImageNet.pth.tar')['state_dict'])
        current_epoch = 0
        # LEARNING_RATE = 0.5*1e-4
        
    # bestLoss = 1e3   
    for epoch in range(current_epoch, EPOCHS):
        DeepJSCC_V.train()
        print('========================')
        print('lr:%.4e'%optimizer.param_groups[0]['lr'])    
        

        # if epoch == 40:
        #     optimizer.param_groups[0]['lr'] =  0.5*1e-4
            
        # Model training
        for i, (x_input, _) in enumerate(train_loader):
            # print(i)%
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
            
            # if i % SAVE_RREQ == 0:
            #     if not os.path.exists('./JSCC_models'): 
            #         os.makedirs('./JSCC_models')
            #     torch.save({'state_dict': DeepJSCC_V.state_dict(), }, './JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_ImageNet.pth.tar')
            #     print('Model saved')
        
        # Model Evaluation
        DeepJSCC_V.eval()
        totalLoss = 0
        with torch.no_grad():
            for i, (test_input, _) in enumerate(test_loader):
                test_input = test_input.cuda()
                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                CR = 0.1+0.9*torch.rand(test_input.shape[0], 1).cuda()
                test_rec =  DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)
            
                totalLoss += criterion(test_input, test_rec).item() * test_input.size(0) 
            averageLoss = totalLoss / 5000
            print('averageLoss=', averageLoss)
            if averageLoss < bestLoss:
                # Model saving
                if not os.path.exists('./JSCC_models'): 
                    os.makedirs('./JSCC_models')
                torch.save({'state_dict': DeepJSCC_V.state_dict(), }, './JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20_ImageNet.pth.tar')
                print('Model saved')
                bestLoss = averageLoss  
             
    # print('Training for DeepJSCC_V is finished!')































