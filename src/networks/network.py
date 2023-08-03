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
#                   Encoder + Bottleneck
#-----------------------------------------------
# Input: masked image + mask
# Output: latent space
class Encoder_Bottleneck(nn.Module):
    def __init__(self, opt):
        super(Encoder_Bottleneck, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # print('fusion : ', fusion.shape)
        # network forward part
        out = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        # latent output
        return out

#-----------------------------------------------
#                   Encoder + Bottleneck v2
#-----------------------------------------------
# Input: masked image + mask
# Output: latent space
class Encoder_Bottleneckv2(nn.Module):
    def __init__(self, opt):
        super(Encoder_Bottleneckv2, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # print('fusion : ', fusion.shape)
        # network forward part
        out = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        # latent output
        return out

#-----------------------------------------------
#                   Decoder
#-----------------------------------------------
# Input: Latent space
# Output: filled image
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        # Upsampling
        self.up1 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, x):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        # fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # network forward part
        # print('decoder input shape : ',type(x), len(x))
        # print('x y shapes', x.shape, y.shape)
        # print(x[0].shape, x[1].shape)
        out = self.up1(x)                                             # out: batch * 128 * 128 * 128
        out = self.up2(out)                                             # out: batch * 128 * 128 * 128
        out = self.up3(out)                                             # out: batch * 64 * 256 * 256
        out = self.up4(out)                                             # out: batch * 3 * 256 * 256
        # final output
        return out

#-----------------------------------------------
#                   Decoder
#-----------------------------------------------
# Input: Latent space
# Output: Latent space
class Bottleneck(nn.Module):
    def __init__(self, opt):
        super(Bottleneck, self).__init__()
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        
    def forward(self, x):

        out = self.b1(x)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64

        return out

#-----------------------------------------------
#             normal  Encoder + Bottleneck
#-----------------------------------------------

# Input: masked image + mask
# Output: latent space
class Encoder_Bottleneck_basic(nn.Module):
    def __init__(self, opt):
        super(Encoder_Bottleneck_basic, self).__init__()
        # Downsampling
        self.down1 = Conv2dLayer(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        
    def forward(self, img):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        # fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # print('fusion : ', fusion.shape)
        # network forward part
        out = self.down1(img)                                        # out: batch * 64 * 256 * 256
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        # latent output
        return out


#-----------------------------------------------
#             normal  Decoder
#-----------------------------------------------
# Input: Latent space
# Output: filled image
class Decoder_basic(nn.Module):
    def __init__(self, opt):
        super(Decoder_basic, self).__init__()
        # Upsampling
        self.up1 = TransposeConv2dLayer(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeConv2dLayer(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = Conv2dLayer(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, x):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        # fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # network forward part
        # print('decoder input shape : ',type(x), len(x))
        # print('x y shapes', x.shape, y.shape)
        # print(x[0].shape, x[1].shape)
        out = self.up1(x)                                             # out: batch * 128 * 128 * 128
        out = self.up2(out)                                             # out: batch * 128 * 128 * 128
        out = self.up3(out)                                             # out: batch * 64 * 256 * 256
        out = self.up4(out)                                             # out: batch * 3 * 256 * 256
        # final output
        return out




#-----------------------------------------------
#                   FCN
#-----------------------------------------------
# Input: ?
# Output: ?

# class AutoEncoder(nn.Module):
#     def __init__(self, opt):
#         super(AutoEncoder, self).__init__()
#         self.encoder = Encoder_Bottleneck2(opt)
#         self.decoder = Decoder2(opt)

#     def forward(self, img, mask):
#         z = self.encoder( img, mask)
#         return self.decoder(z)

#-----------------------------------------------
#               gated Autoencoder
#-----------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Downsampling
            GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.decoder = nn.Sequential(
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        )

    def forward(self, img, mask):
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)   
        z = self.encoder(fusion)
        return self.decoder(z)


#-----------------------------------------------
#               normal Autoencoder
#-----------------------------------------------

class AutoEncoder_basic(nn.Module):
    def __init__(self, opt):
        super(AutoEncoder_basic, self).__init__()
        self.encoder = nn.Sequential(
            # Downsampling
            Conv2dLayer(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.decoder = nn.Sequential(
            TransposeConv2dLayer(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            TransposeConv2dLayer(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        )

    def forward(self, img):
        # masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        # fusion = torch.cat((masked_img, mask), 1)   
        # print(img.shape)
        z = self.encoder(img)
        return self.decoder(z)




# Output: latent space
class Encoder_Bottleneck2(nn.Module):
    def __init__(self, opt):
        super(Encoder_Bottleneck2, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        out = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        # print('last out encoder : ', out.shape)
        # latent output
        return out

#-----------------------------------------------
#                   Decoder
#-----------------------------------------------
# Input: Latent space
# Output: filled image
class Decoder2(nn.Module):
    def __init__(self, opt):
        super(Decoder2, self).__init__()
        # Upsampling
        self.up1 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, x):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        # fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # network forward part
        # print('decoder input shape : ',type(x), len(x))
        # print('x y shapes', x.shape, y.shape)
        # print(x[0].shape, x[1].shape)
        out = self.up1(x)                                             # out: batch * 128 * 128 * 128
        out = self.up2(out)                                             # out: batch * 128 * 128 * 128
        out = self.up3(out)                                             # out: batch * 64 * 256 * 256
        out = self.up4(out)                                             # out: batch * 3 * 256 * 256
        # final output
        return out



#-----------------------------------------------
#                   AE with skip connections
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class AE(nn.Module):
    def __init__(self, opt):
        super(AE, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Upsampling
        self.up1 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = GatedConv2d(opt.latent_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region

        masked_img = img * (1 - mask) + mask            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)   
        # print('fusion : ', fusion.shape)                    # in: batch * 2 * 256 * 256
        # network forward part
        # print('fusion ', fusion.shape)
        down1 = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        # print('down1 ', down1.shape)
        down2 = self.down2(down1)                                           # out: batch * 128 * 128 * 128
        # print('down2 ', down2.shape)
        down3 = self.down3(down2)                                           # out: batch * 256 * 128 * 128
        # print('down3 ', down3.shape)
        down4 = self.down4(down3)                                           # out: batch * 256 * 64 * 64
        # print('down4 ', down4.shape)
        out = self.b1(down4)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        # dec3 = torch.cat((dec3, enc1), dim=1)
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        # print('last bottleneck ', out.shape)
        # print('out b8 shape', out.shape)
        up1 = self.up1(out)                                             # out: batch * 128 * 128 * 128
        # print('up1 ', out.shape)
        # print('out up1 shape', up1.shape)
        out = torch.cat((up1, down2), dim=1)
        # print('out up1 second shape', out.shape)
        up2 = self.up2(out)                                             # out: batch * 128 * 128 * 128
        # print('out up2 shape', up2.shape)
        up3 = self.up3(up2)                                             # out: batch * 64 * 256 * 256
        out = torch.cat((up3, down1), dim=1)
        # print('out up3 shape', up3.shape)
        up4 = self.up4(out)                                             # out: batch * 3 * 256 * 256
        # print('out up4/final shape', up4.shape)
        # final output
        return up4

# with prints ! :/
class AE2(nn.Module):
    def __init__(self, opt):
        super(AE2, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Upsampling
        self.up1 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        print('masked_img : ', masked_img.shape)
        print('img : ', img.shape)
        print('mask : ', mask.shape)
        fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        print('fusion : ', fusion.shape)
        # network forward part
        out = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        print('out shape down1: ',out.shape)
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        print('out shape down2: ',out.shape)
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        print('out shape down3: ',out.shape)
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        print('out shape down4: ',out.shape)
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b1: ',out.shape)
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b2: ',out.shape)
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b3: ',out.shape)
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b4: ',out.shape)
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b5: ',out.shape)
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b6: ',out.shape)
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b7: ',out.shape)
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        print('out shape b8: ',out.shape)

        out = self.up1(out)                                             # out: batch * 128 * 128 * 128
        print('out shape up1: ',out.shape)
        out = self.up2(out)                                             # out: batch * 128 * 128 * 128
        print('out shape up2: ',out.shape)
        out = self.up3(out)                                             # out: batch * 64 * 256 * 256
        print('out shape up3: ',out.shape)
        out = self.up4(out)         
        print('out shape up4: ',out.shape)
        print('last out encoder : ', out.shape)
        # latent output
        return out, mask

#-----------------------------------------------
#               U Net + Y Net
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image


# class YNet_general(nn.Module):

#     def __init__(self, in_channels=1, out_channels=1, mask_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
#                  cat_merge=True):
#         super(YNet_general, self).__init__()

#         self.ffc = ffc
#         self.skip_ffc = skip_ffc
#         self.ratio_in = ratio_in
#         self.cat_merge = cat_merge

#         features = init_features
#         ############### Regular ##################################
#         self.encoder1 = GatedConv2d(in_channels + mask_channels, init_features, 7, 1, 3, pad_type = "reflect", activation = 'lrelu', norm = 'none')
#         self.encoder2 = GatedConv2d(init_features, init_features * 2, 4, 2, 1, pad_type = "reflect", activation = 'lrelu', norm = 'in')
#         self.encoder3 = GatedConv2d(init_features * 2, init_features * 4, 3, 1, 1, pad_type = "reflect", activation = 'lrelu', norm = 'in')
#         self.encoder4 = GatedConv2d(init_features * 4, init_features * 4, 4, 2, 1, pad_type = "reflect", activation = 'lrelu', norm = 'in')
#         self.encoder5 = GatedConv2d(init_features * 4, init_features * 4, 4, 2, 1, pad_type = "reflect", activation = 'lrelu', norm = 'in')
#         self.encoder6 = GatedConv2d(init_features * 4, init_features * 4, 4, 2, 1, pad_type = "reflect", activation = 'lrelu', norm = 'in')

#         # self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
#         # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.encoder2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
#         # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
#         # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.encoder4 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
#         # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         if ffc:
#             ################ FFC #######################################
#             self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
#             self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
#                                          ratio_gout=ratio_in)  # was 1,2
#             self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
#                                          ratio_gout=ratio_in)
#             self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
#                                          ratio_gout=ratio_in)  # was 8
#             self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

#         else:
#             ############### Regular ##################################
#             self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
#             self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder2_f = YNet_general._block(features, features * 2, name="enc2_2")  # was 1,2
#             self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder3_f = YNet_general._block(features * 2, features * 4, name="enc3_2")  #
#             self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.encoder4_f = YNet_general._block(features * 4, features * 4, name="enc4_2")  # was 8
#             self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

#         if skip_ffc and not ffc:
#             self.upconv4 = nn.ConvTranspose2d(
#                 features * 16, features * 8, kernel_size=2, stride=2  # 16
#             )
#             self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
#             self.upconv3 = nn.ConvTranspose2d(
#                 features * 8, features * 4, kernel_size=2, stride=2
#             )
#             self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
#             self.upconv2 = nn.ConvTranspose2d(
#                 features * 4, features * 2, kernel_size=2, stride=2
#             )
#             self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
#             self.upconv1 = nn.ConvTranspose2d(
#                 features * 2, features, kernel_size=2, stride=2
#             )
#             self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

#         elif skip_ffc and ffc:
#             self.upconv4 = nn.ConvTranspose2d(
#                 features * 16, features * 8, kernel_size=2, stride=2  # 16
#             )
#             self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
#             self.upconv3 = nn.ConvTranspose2d(
#                 features * 8, features * 4, kernel_size=2, stride=2
#             )
#             self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
#             self.upconv2 = nn.ConvTranspose2d(
#                 features * 4, features * 2, kernel_size=2, stride=2
#             )
#             self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
#             self.upconv1 = nn.ConvTranspose2d(
#                 features * 2, features, kernel_size=2, stride=2
#             )
#             self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

#         else:
#             self.upconv4 = nn.ConvTranspose2d(
#                 features * 16, features * 8, kernel_size=2, stride=2  # 16
#             )
#             self.decoder4 = YNet_general._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
#             self.upconv3 = nn.ConvTranspose2d(
#                 features * 8, features * 4, kernel_size=2, stride=2
#             )
#             self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
#             self.upconv2 = nn.ConvTranspose2d(
#                 features * 4, features * 2, kernel_size=2, stride=2
#             )
#             self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
#             self.upconv1 = nn.ConvTranspose2d(
#                 features * 2, features, kernel_size=2, stride=2
#             )
#             self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )
#         self.softmax = nn.Softmax2d()
#         self.catLayer = ConcatTupleLayer()

#     def apply_fft(self, inp, batch):
#         ffted = torch.fft.fftn(inp)
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#         return ffted

#     def forward(self, mask, x):
#         batch = x.shape[0]
#         # enc1 = self.encoder1(x)
#         # enc2 = self.encoder2(self.pool1(enc1))

#         # enc3 = self.encoder3(self.pool2(enc2))

#         # enc4 = self.encoder4(self.pool3(enc3))
#         # enc4_2 = self.pool4(enc4)
#         masked_img = x * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
#         fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
#         print('fusion ', fusion.shape)
#         # network forward part
#         enc1 = self.encoder1(fusion)                                        # out: batch * 32 * 256 * 256
#         print('enc1 : ', enc1.shape)
#         enc2 = self.encoder2(enc1)                                           # out: batch * 64 * 128 * 128
#         print('enc2 : ', enc2.shape)
#         enc3 = self.encoder3(enc2)                                           # out: batch * 128 * 128 * 128
#         print('enc3 : ', enc3.shape)
#         enc4 = self.encoder4(enc3)                                           # out: batch * 128 * 64 * 64
#         print('enc4 : ', enc4.shape)
#         enc5= self.encoder5(enc4)                                           # out: batch * 128 * 32 * 32
#         print('enc5 : ', enc5.shape)
#         enc6= self.encoder6(enc5)                                           # out: batch * 128 * 16 * 16
#         print('enc6 : ', enc6.shape)

#         if self.ffc:
#             enc1_f = self.encoder1_f(x)
#             enc1_l, enc1_g = enc1_f
#             if self.ratio_in == 0:
#                 enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
#             elif self.ratio_in == 1:
#                 enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
#             else:
#                 enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

#             enc2_l, enc2_g = enc2_f
#             if self.ratio_in == 0:
#                 enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
#             elif self.ratio_in == 1:
#                 enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
#             else:
#                 enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

#             enc3_l, enc3_g = enc3_f
#             if self.ratio_in == 0:
#                 enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
#             elif self.ratio_in == 1:
#                 enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
#             else:
#                 enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

#             enc4_l, enc4_g = enc4_f
#             if self.ratio_in == 0:
#                 enc4_f2 = self.pool1_f(enc4_l)
#             elif self.ratio_in == 1:
#                 enc4_f2 = self.pool1_f(enc4_g)
#             else:
#                 enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

#         else:
#             enc1_f = self.encoder1_f(x)
#             enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
#             enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
#             enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
#             enc4_f2 = self.pool4(enc4_f)

#         if self.cat_merge:
#             a = torch.zeros_like(enc6)
#             b = torch.zeros_like(enc4_f2)
#             print('a ', a.shape, 'b ', b.shape)
#             enc6 = enc6.view(torch.numel(enc6), 1)
#             enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)
#             print('enc4_f2 ', enc4_f2.shape)
#             print('enc5 ', enc6.shape)
#             bottleneck = torch.cat((enc6, enc4_f2), 1)
#             bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

#         else:
#             bottleneck = torch.cat((enc6, enc4_f2), 1)

#         bottleneck = self.bottleneck(bottleneck)

#         dec4 = self.upconv4(bottleneck)

#         if self.ffc and self.skip_ffc:
#             enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

#             dec4 = torch.cat((dec4, enc4_in), dim=1)
#             dec4 = self.decoder4(dec4)
#             dec3 = self.upconv3(dec4)

#             enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
#             dec3 = torch.cat((dec3, enc3_in), dim=1)
#             dec3 = self.decoder3(dec3)

#             dec2 = self.upconv2(dec3)
#             enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
#             dec2 = torch.cat((dec2, enc2_in), dim=1)
#             dec2 = self.decoder2(dec2)
#             dec1 = self.upconv1(dec2)
#             enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
#             dec1 = torch.cat((dec1, enc1_in), dim=1)

#         elif self.skip_ffc:
#             enc4_in = torch.cat((enc4, enc4_f), dim=1)

#             dec4 = torch.cat((dec4, enc4_in), dim=1)
#             dec4 = self.decoder4(dec4)
#             dec3 = self.upconv3(dec4)

#             enc3_in = torch.cat((enc3, enc3_f), dim=1)
#             dec3 = torch.cat((dec3, enc3_in), dim=1)
#             dec3 = self.decoder3(dec3)

#             dec2 = self.upconv2(dec3)
#             enc2_in = torch.cat((enc2, enc2_f), dim=1)
#             dec2 = torch.cat((dec2, enc2_in), dim=1)
#             dec2 = self.decoder2(dec2)
#             dec1 = self.upconv1(dec2)
#             enc1_in = torch.cat((enc1, enc1_f), dim=1)
#             dec1 = torch.cat((dec1, enc1_in), dim=1)

#         else:
#             dec4 = torch.cat((dec4, enc4), dim=1)
#             dec4 = self.decoder4(dec4)
#             dec3 = self.upconv3(dec4)
#             dec3 = torch.cat((dec3, enc3), dim=1)
#             dec3 = self.decoder3(dec3)
#             dec2 = self.upconv2(dec3)
#             dec2 = torch.cat((dec2, enc2), dim=1)
#             dec2 = self.decoder2(dec2)
#             dec1 = self.upconv1(dec2)
#             dec1 = torch.cat((dec1, enc1), dim=1)

#         dec1 = self.decoder1(dec1)

#         return self.softmax(self.conv(dec1))

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )


# classic UNet without masks
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        # print("enc1 shape :" , enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        # print("enc2 shape :" , enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        # print("enc3 shape :" , enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
        # print("enc4 shape :" , enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)
        # return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

# masked UNet 

class UNet_masked(nn.Module):

    def __init__(self, opt, in_channels=1, out_channels=1, init_features=32):
        super(UNet_masked, self).__init__()

        features = init_features
        self.opt = opt

        self.encoder1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.encoder2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.encoder3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.encoder4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        # self.bottleneck = UNet_masked._block(features * 8, features * 16, name="bottleneck")
        self.bottle1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # self.bottle2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        # self.upconv4 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

        # self.up2 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # self.up4 = GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

        self.upconv4 = nn.ConvTranspose2d(  features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_masked._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(  features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_masked._block((   features * 6), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(  features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_masked._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(  features * 2, features, kernel_size=2, stride=2 )
        self.decoder1 = UNet_masked._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features*4, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

#         self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):

    def forward(self, x, mask):
        masked_img = x * (1 - mask) + mask  

        fusion = torch.cat((masked_img, mask), 1)   
        # print('fusion : ', fusion.shape)
        enc1 = self.encoder1(fusion)
        # print('enc1 : ', enc1.shape)
        enc2 = self.encoder2(enc1)
        # print('enc2 : ', enc2.shape)
        enc3 = self.encoder3(enc2)
        # print('enc3 : ', enc3.shape)
        enc4 = self.encoder4(enc3)
        # print('enc4 : ', enc4.shape)
        
        bottleneck1 = self.bottle1(enc4)
        # bottleneck2 = self.bottle2(bottleneck1)
        # print('bottleneck : ', bottleneck1.shape)

        dec4 = self.upconv4(bottleneck1)
        # print('dec4 shape first', dec4.shape)
        # print('enc3 shape is.... ',enc3.shape)
        dec4 = torch.cat((dec4, enc3), dim=1)
        # print('dec4 shape second', dec4.shape)
        dec4 = self.decoder4(dec4)
        # print('dec4 : ', dec4.shape)
        dec3 = self.upconv3(dec4)
        # print('dec3 shape first', dec3.shape)
        # print('enc1 shape is.... ',enc1.shape)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec3 = self.decoder3(dec3)
        # print('dec3 : ', dec3.shape)
        # dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc1), dim=1)
        # dec2 = self.decoder2(dec2)
        # print('dec2 : ', dec2.shape)
        # dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)
        # print('dec1 : ', dec1.shape)
        return self.conv(dec3)
        # return self.softmax(self.conv(dec1))

    @staticmethod
    def _block_enc(opt,in_channels, out_channels,kernel_s,stride,padding,dilation, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        GatedConv2d( # opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_s,
                            stride= stride,
                            padding=padding,
                            dilation = dilation ,
                            pad_type = opt.pad,
                            activation = opt.activ_g,
                            norm = opt.norm_g,
                        ),
                    ),
                ]
            )
        )

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 8 * 8
class PatchDiscriminator256(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator256, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                              # out: batch * 64 * 256 * 256
        x = self.block2(x)                                              # out: batch * 128 * 128 * 128
        x = self.block3(x)                                              # out: batch * 256 * 64 * 64
        x = self.block4(x)                                              # out: batch * 256 * 32 * 32
        x = self.block5(x)                                              # out: batch * 256 * 16 * 16
        x = self.block6(x)                                              # out: batch * 1 * 8 * 8
        
        return x

class PatchDiscriminator224(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator224, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 4, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 2, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 2, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 2, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 2, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                              # out: batch * 64 * 256 * 256
        x = self.block2(x)                                              # out: batch * 128 * 128 * 128
        x = self.block3(x)                                              # out: batch * 256 * 64 * 64
        x = self.block4(x)                                              # out: batch * 256 * 32 * 32
        x = self.block5(x)                                              # out: batch * 256 * 16 * 16
        x = self.block6(x)                                              # out: batch * 1 * 8 * 8

        return x


class PatchDiscriminator384(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator384, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        # x = torch.cat((img, mask), 1)
        # print(img.shape)
        x = self.block1(img)                                            # torch.Size([1, 64, 384, 384])  # out: batch * 64 * 256 * 256
        # print(x.shape)
        x = self.block2(x)                                              # torch.Size([1, 128, 192, 192]) # out: batch * 128 * 128 * 128
        # print(x.shape)
        x = self.block3(x)                                              # torch.Size([1, 256, 96, 96]) # out: batch * 256 * 64 * 64
        # print(x.shape)
        x = self.block4(x)                                              # torch.Size([1, 256, 48, 48]) # out: batch * 256 * 32 * 32
        # print(x.shape)
        x = self.block5(x)                                              # torch.Size([1, 256, 24, 24]) # out: batch * 256 * 16 * 16
        # print(x.shape)
        x = self.block6(x)                                              # torch.Size([1, 1, 12, 12]) # out: batch * 1 * 8 * 8 #

        return x


'''
x :  torch.Size([1, 2, 224, 224])
x :  torch.Size([1, 64, 224, 224])
x :  torch.Size([1, 128, 112, 112])
x :  torch.Size([1, 256, 56, 56])
x :  torch.Size([1, 256, 28, 28])
x :  torch.Size([1, 256, 14, 14])
x :  torch.Size([1, 1, 7, 7])

x :  torch.Size([1, 2, 224, 224])
x :  torch.Size([1, 64, 222, 222])
x :  torch.Size([1, 128, 111, 111])
x :  torch.Size([1, 256, 55, 55])
x :  torch.Size([1, 256, 27, 27])
x :  torch.Size([1, 256, 13, 13])
x :  torch.Size([1, 1, 6, 6])
'''
# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        
        x = self.features(x)
        # print(x.shape) # torch.Size([1, 512, 48, 48])
        return x



