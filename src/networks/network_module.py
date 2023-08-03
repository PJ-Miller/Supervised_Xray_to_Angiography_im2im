################ imports ################
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, sn = sn)
        )
    
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out

#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

#-----------------------------------------------
#                Partial ConvBlock
#-----------------------------------------------
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    #              padding=0, dilation=1, groups=1, bias=True):
# class PartialConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
#         super(PartialConv2d, self).__init__()
#                 # Initialize the padding scheme
#         if pad_type == 'reflect':
#             self.pad = nn.ReflectionPad2d(padding)
#         elif pad_type == 'replicate':
#             self.pad = nn.ReplicationPad2d(padding)
#         elif pad_type == 'zero':
#             self.pad = nn.ZeroPad2d(padding)
#         else:
#             assert 0, "Unsupported padding type: {}".format(pad_type)
        
#         # Initialize the normalization type
#         if norm == 'bn':
#             self.norm = nn.BatchNorm2d(out_channels)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm2d(out_channels)
#         elif norm == 'ln':
#             self.norm = LayerNorm(out_channels)
#         elif norm == 'none':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)
        
#         # Initialize the activation funtion
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace = True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace = True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace = True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)

#         self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                                     stride, padding, dilation, groups, bias)
#         self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                                    stride, padding, dilation, groups, False)
#         self.input_conv.apply(weights_init('kaiming'))

#         torch.nn.init.constant_(self.mask_conv.weight, 1.0)

#         # mask is not updated
#         for param in self.mask_conv.parameters():
#             param.requires_grad = False

#     def forward(self, input, mask):
#         # http://masc.cs.gmu.edu/wiki/partialconv
#         # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
#         # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

#         output = self.input_conv(input * mask)
#         if self.input_conv.bias is not None:
#             output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
#                 output)
#         else:
#             output_bias = torch.zeros_like(output)

#         with torch.no_grad():
#             output_mask = self.mask_conv(mask)

#         no_update_holes = output_mask == 0
#         mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

#         output_pre = (output - output_bias) / mask_sum + output_bias
#         output = output_pre.masked_fill_(no_update_holes, 0.0)

#         new_mask = torch.ones_like(output)
#         new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

#         return output, new_mask

# class PartialConv2d(nn.Conv2d):
    
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
#         super(PartialConv2d, self).__init__()
#                 # Initialize the padding scheme
#         if pad_type == 'reflect':
#             self.pad = nn.ReflectionPad2d(padding)
#         elif pad_type == 'replicate':
#             self.pad = nn.ReplicationPad2d(padding)
#         elif pad_type == 'zero':
#             self.pad = nn.ZeroPad2d(padding)
#         else:
#             assert 0, "Unsupported padding type: {}".format(pad_type)
        
#         # Initialize the normalization type
#         if norm == 'bn':
#             self.norm = nn.BatchNorm2d(out_channels)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm2d(out_channels)
#         elif norm == 'ln':
#             self.norm = LayerNorm(out_channels)
#         elif norm == 'none':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)
        
#         # Initialize the activation funtion
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace = True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace = True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace = True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)

#         # whether the mask is multi-channel or not
#         self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
#         self.return_mask = True

#         # generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
#         self.in_channels = in_channels        
#         self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

#         self.last_size = (None, None)
#         self.update_mask = None
#         self.mask_ratio = None

#     def forward(self, input, mask=None):

#         if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
#             self.last_size = (input.data.shape[2], input.data.shape[3])

#             with torch.no_grad():
#                 if self.weight_maskUpdater.type() != input.type():
#                     self.weight_maskUpdater = self.weight_maskUpdater.to(input)

#                 if mask is None:
#                     mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
#                                         input.data.shape[3]).to(input)

#                 self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
#                                             padding=self.padding, dilation=self.dilation, groups=1)

#                 self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
#                 # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
#                 self.update_mask = torch.clamp(self.update_mask, 0, 1)
#                 self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

#         if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
#             self.update_mask.to(input)
#             self.mask_ratio.to(input)

#         raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)

#         if self.bias is not None:
#             bias_view = self.bias.view(1, self.out_channels, 1, 1)
#             output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
#             output = torch.mul(output, self.update_mask)
#         else:
#             output = torch.mul(raw_out, self.mask_ratio)

#         if self.return_mask:
#             return output, self.update_mask
#         else:
#             return output

# class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x


#-----------------------------------------------
#                Gated ConvBlock v2
#-----------------------------------------------
class GatedConv2dv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2dv2, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
