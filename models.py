

import numpy as np
import torch.nn as nn
import torch

from GDN import GDN

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block, self).__init__()
        self.deconv = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv(x)
        out = self.bn(out)
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out   
    
class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr):
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        if snr.shape[0]>1:
            snr = snr.squeeze()
        snr = snr.unsqueeze(1)  
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out


class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    def forward(self, x): 
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out+x
        out = self.prelu(out)
        return out 


class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out+x
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out 
        
# The Encoder model with attention feature blocks
class Encoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(Encoder, self).__init__()    
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv//2
        padding_L = (kernel_sz-1)//2
        self.conv1 = conv_ResBlock(3, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv2 = conv_ResBlock(Nc_conv, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv3 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv4 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv5 = conv_ResBlock(Nc_conv, enc_N, use_conv1x1=True, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.AF1 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF4 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF5 = AF_block(enc_N, enc_N//2, enc_N)
        self.flatten = nn.Flatten() 
    def forward(self, x, snr):    
        out = self.conv1(x) 
        out = self.AF1(out, snr)
        out = self.conv2(out)   
        out = self.AF2(out, snr)
        out = self.conv3(out) 
        out = self.AF3(out, snr)
        out = self.conv4(out) 
        out = self.AF4(out, snr)
        out = self.conv5(out) 
        out = self.AF5(out, snr)
        out = self.flatten(out)
        return out

# The Decoder model with attention feature blocks
class Decoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0]//2
        Nh_AF = Nc_deconv//2
        padding_L = (kernel_sz-1)//2
        self.deconv1 = deconv_ResBlock(self.enc_shape[0], Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv2 = deconv_ResBlock(Nc_deconv, Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv3 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv4 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv5 = deconv_ResBlock(Nc_deconv, 3, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.AF1 = AF_block(self.enc_shape[0], Nh_AF1, self.enc_shape[0])
        self.AF2 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF3 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF4 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF5 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
    def forward(self, x, snr):  
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.AF1(out, snr)
        out = self.deconv1(out) 
        out = self.AF2(out, snr)
        out = self.deconv2(out) 
        out = self.AF3(out, snr)
        out = self.deconv3(out)
        out = self.AF4(out, snr)
        out = self.deconv4(out)
        out = self.AF5(out, snr)
        out = self.deconv5(out, 'sigmoid')  
        return out


# # The complexities of the following Encoder and Decoder models are smaller
# class Encoder(nn.Module):
#     def __init__(self, enc_shape, kernel_sz, Nc_conv):
#         super(Encoder, self).__init__()    
#         enc_N = enc_shape[0]
#         Nh_AF = Nc_conv//2
#         padding_L = (kernel_sz-1)//2
#         self.conv1 = conv_ResBlock(3, Nc_conv//2, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
#         self.conv2 = conv_ResBlock(Nc_conv//2, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
#         self.conv3 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
#         self.conv4 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
#         self.conv5 = conv_ResBlock(Nc_conv, enc_N, use_conv1x1=True, kernel_size = kernel_sz, stride = 1, padding=padding_L)
#         self.AF1 = AF_block(Nc_conv//2, Nh_AF//2, Nc_conv//2)
#         self.AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
#         self.AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)
#         self.AF4 = AF_block(Nc_conv, Nh_AF, Nc_conv)
#         self.AF5 = AF_block(enc_N, enc_N//2, enc_N)
#         self.flatten = nn.Flatten() 
#     def forward(self, x, snr):    
#         out = self.conv1(x) 
#         out = self.AF1(out, snr)
#         out = self.conv2(out)   
#         out = self.AF2(out, snr)
#         out = self.conv3(out) 
#         out = self.AF3(out, snr)
#         out = self.conv4(out) 
#         out = self.AF4(out, snr)
#         out = self.conv5(out) 
#         out = self.AF5(out, snr)
#         out = self.flatten(out)
#         return out

# class Decoder(nn.Module):
#     def __init__(self, enc_shape, kernel_sz, Nc_deconv):
#         super(Decoder, self).__init__()
#         self.enc_shape = enc_shape
#         Nh_AF1 = enc_shape[0]//2
#         Nh_AF = Nc_deconv//2
#         padding_L = (kernel_sz-1)//2
#         self.deconv1 = deconv_ResBlock(self.enc_shape[0], Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
#         self.deconv2 = deconv_ResBlock(Nc_deconv, Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
#         self.deconv3 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
#         self.deconv4 = deconv_ResBlock(Nc_deconv, Nc_deconv//2, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)
#         self.deconv5 = deconv_ResBlock(Nc_deconv//2, 3, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)
#         self.AF1 = AF_block(self.enc_shape[0], Nh_AF1, self.enc_shape[0])
#         self.AF2 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
#         self.AF3 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
#         self.AF4 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
#         self.AF5 = AF_block(Nc_deconv//2, Nh_AF//2, Nc_deconv//2)
#     def forward(self, x, snr):  
#         out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
#         out = self.AF1(out, snr)
#         out = self.deconv1(out) 
#         out = self.AF2(out, snr)
#         out = self.deconv2(out) 
#         out = self.AF3(out, snr)
#         out = self.deconv3(out)
#         out = self.AF4(out, snr)
#         out = self.deconv4(out)
#         out = self.AF5(out, snr)
#         out = self.deconv5(out, 'sigmoid')  
#         return out



# Power normalization before transmission
# Note: if P = 1, the symbol power is 2
# If you want to set the average power as 1, please change P as P=1/np.sqrt(2)
def Power_norm(z, P = 1):
    batch_size, z_dim = z.shape
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1)
    return np.sqrt(P*z_dim)*z/z_M.t()

def Power_norm_complex(z, P = 1): 
    batch_size, z_dim = z.shape
    z_com = torch.complex(z[:, 0:z_dim:2], z[:, 1:z_dim:2])
    z_com_conj = torch.complex(z[:, 0:z_dim:2], -z[:, 1:z_dim:2])
    z_power = torch.sum(z_com*z_com_conj, 1).real
    z_M = z_power.repeat(z_dim//2, 1)
    z_nlz = np.sqrt(P*z_dim)*z_com/torch.sqrt(z_M.t())
    z_out = torch.zeros(batch_size, z_dim).cuda()
    z_out[:, 0:z_dim:2] = z_nlz.real
    z_out[:, 1:z_dim:2] = z_nlz.imag
    return z_out

# The (real) AWGN channel    
def AWGN_channel(x, snr, P = 2):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P/gamma)*torch.randn(batch_size, length).cuda()
    y = x+noise
    return y

def AWGN_complex(x, snr, Ps = 1):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    n_I = torch.sqrt(Ps/gamma)*torch.randn(batch_size, length).cuda()
    n_R = torch.sqrt(Ps/gamma)*torch.randn(batch_size, length).cuda()
    noise = torch.complex(n_I, n_R)
    y = x + noise
    return y

# Please set the symbol power if it is not a default value
def Fading_channel(x, snr, P = 2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    h_I = torch.randn(batch_size, K).cuda()
    h_R = torch.randn(batch_size, K).cuda() 
    h_com = torch.complex(h_I, h_R)  
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com*x_com
    
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    noise = torch.complex(n_I, n_R)
    
    y_add = y_com + noise
    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).cuda()
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out



# Note: if P = 1, the symbol power is 2
# If you want to set the average power as 1, please change P as P=1/np.sqrt(2)
def Power_norm_VLC(z, cr, P = 1):
    batch_size, z_dim = z.shape
    Kv = torch.ceil(z_dim*cr).int()
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1).cuda()
    return torch.sqrt(Kv*P)*z/z_M.t()


def AWGN_channel_VLC(x, snr, cr, P = 2):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    mask = mask_gen(length, cr).cuda()
    noise = torch.sqrt(P/gamma)*torch.randn(1, length).cuda()
    noise = noise*mask
    y = x+noise
    return y


def Fading_channel_VLC(x, snr, cr, P = 2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    mask = mask_gen(K, cr).cuda()
    h_I = torch.randn(batch_size, K).cuda()
    h_R = torch.randn(batch_size, K).cuda() 
    h_com = torch.complex(h_I, h_R)  
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com*x_com
    
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    noise = torch.complex(n_I, n_R)*mask
    
    y_add = y_com + noise
    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).cuda()
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


def Channel(z, snr, channel_type = 'AWGN'):
    z = Power_norm(z)
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    return z


def Channel_VLC(z, snr, cr, channel_type = 'AWGN'):
    z = Power_norm_VLC(z, cr)
    if channel_type == 'AWGN':
        z = AWGN_channel_VLC(z, snr, cr)
    elif channel_type == 'Fading':
        z = Fading_channel_VLC(z, snr, cr)
    return z


def mask_gen(N, cr, ch_max = 48):
    MASK = torch.zeros(cr.shape[0], N).int()
    nc = N//ch_max
    for i in range(0, cr.shape[0]):
        L_i = nc*torch.round(ch_max*cr[i]).int()
        MASK[i, 0:L_i] = 1
    return MASK


class ADJSCC(nn.Module):
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(ADJSCC, self).__init__()
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        self.decoder = Decoder(enc_shape, Kernel_sz, Nc) 
    def forward(self, x, snr, channel_type = 'AWGN'):
        z = self.encoder(x, snr)
        z = Channel(z, snr, channel_type)
        out = self.decoder(z, snr)
        return out 

# The DeepJSCC_V model, also called ADJSCC_V
class ADJSCC_V(nn.Module):
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(ADJSCC_V, self).__init__()
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        self.decoder = Decoder(enc_shape, Kernel_sz, Nc) 
    def forward(self, x, snr, cr, channel_type = 'AWGN'):
        z = self.encoder(x, snr)  
        z = z*mask_gen(z.shape[1], cr).cuda()
        z = Channel_VLC(z, snr, cr, channel_type)
        out = self.decoder(z, snr)
        return out



















