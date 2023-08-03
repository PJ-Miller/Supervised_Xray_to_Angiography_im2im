################ imports ################
import torch
import torch.nn as nn
import torch.nn.init as init
# from .ffc import FFC_BN_ACT, ConcatTupleLayer
from .network_module import *
from collections import OrderedDict


def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


#-----------------------------------------------
#            original Bottleneck
#-----------------------------------------------
# 5 layers gated convolutions,  kernelsize 3,3
# b2, b3 dilation 2, 4
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_original(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_original, self).__init__()
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):
        
        out = self.b1(x)                                                # out: batch * latent_multi * 64 * 64
        out = self.b2(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b3(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b4(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b5(out)                                              # out: batch * 4 * 64 * 64

        return out



#-----------------------------------------------
#            basic original Bottleneck
#-----------------------------------------------
# 5 layers convolutions,  kernelsize 3,3
# b2, b3 dilation 2, 4
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_original_basic(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_original_basic, self).__init__()
        # Bottleneck
        self.b1 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):
        
        out = self.b1(x)                                                # out: batch * latent_multi * 64 * 64
        out = self.b2(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b3(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b4(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b5(out)                                              # out: batch * 4 * 64 * 64

        return out


#-----------------------------------------------
#            bigger Bottleneck
#-----------------------------------------------
# 7 layers  convolutions,  kernelsize 3,3
# b2, b3, b4, b5 dilation 2, 4, 8, 16
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_bigger_basic(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_bigger_basic, self).__init__()
        # Bottleneck
        self.b1 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = Conv2dLayer(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):

        out = self.b1(x)                                                # out: batch * latent_multi * 64 * 64
        out = self.b2(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b3(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b4(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b5(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b6(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64

        return out




#-----------------------------------------------
#            smaller Bottleneck
#-----------------------------------------------
# 3 layers gated convolutions,  kernelsize 3,3
# no dilation
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_smaller(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_smaller, self).__init__()
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):

        out = self.b1(x)                                                # out: batch * latent_multi * 64 * 64
        out = self.b2(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64

        return out


#-----------------------------------------------
#            bigger Bottleneck
#-----------------------------------------------
# 7 layers gated convolutions,  kernelsize 3,3
# b2, b3, b4, b5 dilation 2, 4, 8, 16
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_bigger(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_bigger, self).__init__()
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):

        out = self.b1(x)                                                # out: batch * latent_multi * 64 * 64
        out = self.b2(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b3(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b4(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b5(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b6(out)                                              # out: batch * latent_multi * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64

        return out



#-----------------------------------------------
#      autoencoder-like bigger Bottleneck
#-----------------------------------------------
# 7 layers gated convolutions, 1 layer Transpose gated convolution ,  kernelsize 3,3
# b2, b3 dilation 2, 4
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_Bigger_aeLike(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_Bigger_aeLike, self).__init__()
        # Bottleneck
        self.down1  = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b1     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up1    = TransposeGatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2    = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

    def forward(self, x):

        out = self.down1(x)                                                 # out: latent_multi * 256 * 32 * 32
        out = self.b1(out)                                                  # out: latent_multi * 256 * 32 * 32
        out = self.b2(out)                                                  # out: latent_multi * 256 * 32 * 32
        out = self.b3(out)                                                  # out: latent_multi * 256 * 32 * 32
        out = self.b4(out)                                                  # out: latent_multi * 256 * 32 * 32
        out = self.b5(out)                                                  # out: latent_multi * 256 * 32 * 32
        out = self.up1(out)                                                 # out: latent_multi * 256 * 64 * 64
        out = self.up2(out)                                                 # out: batch * 256 * 64 * 64

        return out



#-----------------------------------------------
#      reverse-autoencoder-like Bottleneck
#-----------------------------------------------
# 7 layers gated convolutions,  kernelsize 3,3
# b2, b3, b4, b5 dilation 2, 4, 8, 16
# Input: Latent space       64 * 64
# Output: Latent space      64 * 64
class Bottleneck_Reverse_aeLike(nn.Module):
    def __init__(self, opt):
        super(Bottleneck_Reverse_aeLike, self).__init__()
        # Bottleneck
        self.up1    = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2    = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b1     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5     = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * opt.latent_multi, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down1  = GatedConv2d(opt.latent_channels * opt.latent_multi, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
       
    def forward(self, x):

        out = self.up1(x)                                               # out: batch * latent_multi * 128 * 128
        out = self.up2(out)                                             # out: batch * latent_multi * 128 * 128
        out = self.b1(out)                                              # out: batch * latent_multi * 128 * 128
        out = self.b2(out)                                              # out: batch * latent_multi * 128 * 128
        out = self.b3(out)                                              # out: batch * latent_multi * 128 * 128
        out = self.b4(out)                                              # out: batch * latent_multi * 128 * 128
        out = self.b5(out)                                              # out: batch * latent_multi * 128 * 128
        out = self.down1(out)                                           # out: batch * 256 * 64 * 64

        return out

