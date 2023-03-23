

import torch
import torch.nn as nn
import os

from utils import *
from models import *


BATCH_SIZE = 128
EPOCHS = 200 
LEARNING_RATE = 1e-4
PRINT_RREQ = 250


CHANNEL = 'AWGN'  # Choose AWGN or Fading
IMG_SIZE = [3, 32, 32]
N_channels = 256
Kernel_sz = 5

# Parameter enc_out_shape[0] specifies the compresison ratio
enc_out_shape = [32, IMG_SIZE[1]//4, IMG_SIZE[2]//4]

CR = 96//enc_out_shape[0]

x_train, x_test = Load_cifar10_data()
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

current_epoch = 0
CONTINUE_TRAINING = False	

KSZ = str(Kernel_sz)+'x'+str(Kernel_sz)+'_'
if __name__ == '__main__':

    DeepJSCC = ADJSCC(enc_out_shape, Kernel_sz, N_channels).cuda()
    # DeepJSCC = nn.DataParallel(DeepJSCC)
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(DeepJSCC.parameters(), lr=LEARNING_RATE)
    
    bestLoss = 1e3
    if CONTINUE_TRAINING == True:
        DeepJSCC.load_state_dict(torch.load('./JSCC_models/DeepJSCC_'+KSZ+CHANNEL+'_'+str(CR)+'_'+str(N_channels)+'.pth.tar')['state_dict'])
        current_epoch = 0
        bestLoss = 1
        
        
    for epoch in range(current_epoch, EPOCHS):
        DeepJSCC.train()
        print('========================')
        print('lr:%.4e'%optimizer.param_groups[0]['lr'])     
        
        # Model training
        for i, x_input in enumerate(train_loader):
            x_input = x_input.cuda()
            
            SNR_TRAIN = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
            x_rec = DeepJSCC(x_input, SNR_TRAIN, CHANNEL)
            loss = criterion(x_input, x_rec) 
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % PRINT_RREQ == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
                
        # Model Evaluation
        DeepJSCC.eval()
        totalLoss = 0
        with torch.no_grad():
            for i, test_input in enumerate(test_loader):
                test_input = test_input.cuda()
                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                test_rec = DeepJSCC(test_input, SNR_TEST, CHANNEL)
                totalLoss += criterion(test_rec, test_input).item() * test_input.size(0)  
            averageLoss = totalLoss / (len(test_dataset))
            print('averageLoss=', averageLoss)
            if averageLoss < bestLoss:
                # Model saving
                if not os.path.exists('./JSCC_models'): 
                    os.makedirs('./JSCC_models')
                torch.save({'state_dict': DeepJSCC.state_dict(), }, './JSCC_models/DeepJSCC_'+KSZ+CHANNEL+'_'+str(CR)+'_'+str(N_channels)+'.pth.tar')
                print('Model saved')
                bestLoss = averageLoss

    print('Training for DeepJSCC_'+str(CR)+' is finished!')































