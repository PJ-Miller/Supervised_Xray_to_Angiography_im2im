################ imports ################
from .utils_core import logg
import torch
import numpy as np

from ..networks import network, network_bottlenecks

from .utils_core import * 
# import utils


# SET SEED
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_encoder(opt):
    # Initialize the networks
    encoder = network.Encoder_Bottleneck(opt)

    logg(f'The Encoder + Bottleneck is created!')
    
    # Init the networks
    if opt.finetune_path_encoder:
        # logg(f'this path {opt.finetune_path_encoder} exists : {os.path.exists(opt.finetune_path_encoder)}')
        pretrained_net = torch.load(opt.finetune_path_encoder)
        encoder = load_dict(encoder, pretrained_net)
        logg('Load generator_encoder with %s' % opt.finetune_path_encoder)
    else:
        network.weights_init(encoder, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return encoder

def create_encoder_basic(opt):
    # Initialize the networks
    encoder = network.Encoder_Bottleneck_basic(opt)

    logg(f'The Encoder + Bottleneck conv version is created!')
    
    # Init the networks
    if opt.finetune_path_encoder:
        # logg(f'this path {opt.finetune_path_encoder} exists : {os.path.exists(opt.finetune_path_encoder)}')
        pretrained_net = torch.load(opt.finetune_path_encoder)
        encoder = load_dict(encoder, pretrained_net)
        logg('Load generator_encoder with %s' % opt.finetune_path_encoder)
    else:
        network.weights_init(encoder, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return encoder


def create_decoder(opt):
    # Initialize the networks
    decoder = network.Decoder(opt)

    logg(f'The Decoder is created!')
    
    # Init the networks
    if opt.finetune_path_decoder:
        pretrained_net = torch.load(opt.finetune_path_decoder)
        decoder = load_dict(decoder, pretrained_net)
        logg('Load generator_decoder with %s' % opt.finetune_path_decoder)
    else:
        network.weights_init(decoder, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return decoder

def create_decoder_basic(opt):
    # Initialize the networks
    decoder = network.Decoder_basic(opt)

    logg(f'The Decoder conv version is created!')
    
    # Init the networks
    if opt.finetune_path_decoder:
        pretrained_net = torch.load(opt.finetune_path_decoder)
        decoder = load_dict(decoder, pretrained_net)
        logg('Load generator_decoder with %s' % opt.finetune_path_decoder)
    else:
        network.weights_init(decoder, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return decoder



def create_bottleneck(opt):
    # Initialize the networks

    # bottleneck = network_bottlenecks.Bottleneck_smaller(opt,latent_multi=2)
    bottleneck = network_bottlenecks.Bottleneck_original(opt)

    logg(f'Additional bottleneck created!')
    
    # Init the networks
    if opt.finetune_path_bottleneck:
        pretrained_net = torch.load(opt.finetune_path_bottleneck)
        bottleneck = load_dict(bottleneck, pretrained_net)
        logg('Load generator_bottleneck with %s' % opt.finetune_path_bottleneck)
    else:
        network.weights_init(bottleneck, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return bottleneck


def create_bottleneck_basic(opt):
    # Initialize the networks

    # bottleneck = network_bottlenecks.Bottleneck_smaller(opt,latent_multi=2)
    bottleneck = network_bottlenecks.Bottleneck_original_basic(opt) # Bottleneck_bigger_basic

    logg(f'Additional bottleneck conv version created!')
    
    # Init the networks
    if opt.finetune_path_bottleneck:
        pretrained_net = torch.load(opt.finetune_path_bottleneck)
        bottleneck = load_dict(bottleneck, pretrained_net)
        logg('Load generator_bottleneck with %s' % opt.finetune_path_bottleneck)
    else:
        network.weights_init(bottleneck, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return bottleneck


def create_bottleneck_basic2(opt):
    # Initialize the networks

    # bottleneck = network_bottlenecks.Bottleneck_smaller(opt,latent_multi=2)
    bottleneck = network_bottlenecks.Bottleneck_bigger_basic(opt) # Bottleneck_bigger_basic

    logg(f'Additional bottleneck conv version created!')
    
    # Init the networks
    if opt.finetune_path_bottleneck:
        pretrained_net = torch.load(opt.finetune_path_bottleneck)
        bottleneck = load_dict(bottleneck, pretrained_net)
        logg('Load generator_bottleneck with %s' % opt.finetune_path_bottleneck)
    else:
        network.weights_init(bottleneck, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return bottleneck

def create_AE(opt):
    # Initialize the networks
    ae = network.AE(opt)

    logg(f'The AutoEncoder is created!')
    
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        ae = load_dict(ae, pretrained_net)
        logg('Load generator_AE with %s' % opt.finetune_path)
    else:
        network.weights_init(ae, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return ae

def create_AE2(opt):
    # Initialize the networks
    ae = network.AutoEncoder(opt)

    logg(f'The AutoEncoder2 is created!')
    
    # Init the networks
    if opt.finetune_path_autoencoder:
        pretrained_net = torch.load(opt.finetune_path_autoencoder)
        ae = load_dict(ae, pretrained_net)
        logg('Load generator_AE2 with %s' % opt.finetune_path_autoencoder)
    else:
        network.weights_init(ae, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return ae


def create_AEbasic(opt):
    # Initialize the networks
    ae = network.AutoEncoder_basic(opt)

    logg(f'The basic autoencoder is created!')
    
    # Init the networks
    if opt.finetune_path_autoencoder:
        pretrained_net = torch.load(opt.finetune_path_autoencoder)
        ae = load_dict(ae, pretrained_net)
        logg('Load generator_AE2 with %s' % opt.finetune_path_autoencoder)
    else:
        network.weights_init(ae, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return ae

def create_Unet(opt):
    # Initialize the networks
    ae = network.UNet_masked(opt)

    logg(f'Unet :D is created!')
    
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        ae = load_dict(ae, pretrained_net)
        logg('Load generator with %s' % opt.finetune_path)
    else:
        network.weights_init(ae, init_type = opt.init_type, init_gain = opt.init_gain)
        logg('Initialize generator with %s type' % opt.init_type)
    return ae



def create_generator(opt):
    # Initialize the networks

    if opt.model == 0:
        generator = network.Inpainting_Network_original(opt)
    elif opt.model ==1:
        generator = network.Inpainting_Network_Enlarged_EncoderDecoder(opt)
    elif opt.model ==2:
        generator = network.Inpainting_Network_Enlarged_EncoderDecoderBottleneck(opt)
    elif opt.model ==3:
        generator = network.Inpainting_Network_Enlarged_Encoder(opt)
    elif opt.model ==4:
        generator = network.Inpainting_Network_Enlarged_Decoder(opt)
    else:
        logg("THIS MODEL DOES NOT EXIST YET!")
        quit()

    logg(f'Generator {opt.model} is created!')
    
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % opt.finetune_path)
    else:
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator224(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator224(opt)
    logg('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    logg('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_discriminator256(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator256(opt)
    logg('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    logg('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_discriminator384(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator384(opt)
    logg('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    logg('Initialize discriminator with %s type' % opt.init_type)
    return discriminator


# def create_perceptualnet():
#     # Pre-trained VGG-16
#     # vgg16 = tv.models.vgg16(pretrained=True)
#     vgg16 = torch.load('vgg16_pretrained.pth')
#     # Get the first 16 layers of vgg16, which is conv3_3
#     perceptualnet = network.PerceptualNet()
#     # Update the parameters
#     load_dict(perceptualnet, vgg16)
#     # It does not gradient
#     for param in perceptualnet.parameters():
#         param.requires_grad = False
#     return perceptualnet

def create_perceptualnet(opt):
    # Pre-trained VGG-16
    # vgg16 = tv.models.vgg16(pretrained=True)
    # opt.path_vgg_pretrained
    vgg16 = torch.load(opt.path_vgg_pretrained)
    # vgg16 = torch.load('vgg16_pretrained.pth')
    # Get the first 16 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Update the parameters
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
  