# import argparse
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# import network
# import dataset
# import os
# import torch.nn as nn
# import focal_frequency_loss
# import utils
# import logging
# import pyiqa
# import torchvision
# from utils import logg

################ imports ################
import torch
import numpy as np
# import random
from ..utils.utils_core import logg
# import torch.backends.cudnn as cudnn
from ..utils import utils_core, utils_networks, utils_sample
import torch.nn as nn
from ..metrics import focal_frequency_loss
import os
import torchvision
from ..dataloaders import dataset
import pyiqa
# import time
from torch.utils.data import DataLoader
# import datetime



'''
python .\validation.py --model 0 --load_name  'D:\\Thesis_Repo\\Documentation\\hello_world_DL_dataset_models\\model_0\\deepfillv3_L1_L1_free\\Model_0_L1_L1_free_form_epoch220_batchsize1.pth' --mask_type free_form --bbox_shape 60  --mask_num 30 --margin 0 --max_angle 4 --max_len 50 --max_width 8 --baseroot D:\\Thesis_Repo\\data\\Shepp_Logan_Ghost\\Hello_World_Deep_Learning_SIIM\\train\\chst
'''

# # SET SEED
# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True



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

def Validation_Prototype_1(opt):
   
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    # testset = dataset.InpaintDataset(opt)
    # print('The overall number of images equals to %d' % len(testset))

    # # Define the dataloader
    # dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    if opt.in_channels == 1:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(1), torchvision.transforms.CenterCrop(opt.imgsize)
                            ])
    else:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(opt.imgsize)
                                ])

    trainset = dataset.InpaintDataset_pairs(opt, transforms=transforms)

    logg('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
   

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    # print(os.path.exists(opt.load_name))
    # pretrained_net = torch.load(opt.load_name)

    # Build networks
    encoder_bottle_net = utils_networks.create_encoder(opt)
    decoder_net = utils_networks.create_decoder(opt)
    bottleneck_net = utils_networks.create_bottleneck(opt)
    # ae_net = utils.create_AE(opt)
    ae_net2 = utils_networks.create_AE2(opt)
    # unet = utils.create_Unet()

    # To device
    if opt.multi_gpu == True:
        encoder_bottle_net = nn.DataParallel(encoder_bottle_net)
        decoder_net = nn.DataParallel(decoder_net)
        bottleneck_net = nn.DataParallel(bottleneck_net)
        ae_net2 = nn.DataParallel(ae_net2)
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()
    else:
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()


    # model = load_dict(generator, pretrained_net)
    pytorch_total_params = sum(p.numel() for p in ae_net2.parameters())
    logg('ae_net2',pytorch_total_params)
    # print('Load generator with %s' % opt.finetune_path)
    # logging.info(model)
    # model = torch.load(opt.load_name)
    # model.cuda()

    logg("----------- LIBRARY NEW METRICS -------------")
    logg("---------------------------------------------")

    iqa_metric_lpips = pyiqa.create_metric('lpips', device=torch.device('cuda'))
    logg(f"lower_better is : {iqa_metric_lpips.lower_better}")

    # iqa_metric_fid = pyiqa.create_metric('fid', device=torch.device('cuda'))
    # logging.info(f"lower_better is : {iqa_metric_fid.lower_better}")

    iqa_metric_psnr = pyiqa.create_metric('psnr', device=torch.device('cuda'))
    logg(f"lower_better is : {iqa_metric_psnr.lower_better}")

    iqa_metric_ssim = pyiqa.create_metric('ssim', device=torch.device('cuda'))
    logg(f"lower_better is : {iqa_metric_ssim.lower_better}")
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='gray')

    # Loss functions
    L1_Loss = nn.L1Loss()
    L2_Loss = nn.MSELoss()
    Huber_Loss = nn.HuberLoss()
    FocalFrequency_Loss = focal_frequency_loss.FocalFrequencyLoss()

    L1_list_with2with = []
    L1_list_without2with = []

    L2_list_with2with = []
    L2_list_without2with = []

    Hub_list_with2with = []
    Hub_list_without2with = []

    Focal_list_with2with = []
    Focal_list_without2with = []

    lpips_list_with2with = []
    lpips_list_without2with = []

    psnr_list_with2with = []
    psnr_list_without2with = []

    ssim_list_with2with = []
    ssim_list_without2with = []

    for batch_idx, (grayscale_without, grayscale_with, mask) in enumerate(dataloader):

        grayscale_with = grayscale_with.cuda()
        grayscale_without = grayscale_without.cuda()
        mask = mask.cuda()

        # inference
        out_enc_net1 = encoder_bottle_net(grayscale_without, mask)  # grayscale_without
        out_bottle =  bottleneck_net(out_enc_net1) 
        out_decoder = ae_net2.decoder(out_bottle)   
        out_ae = ae_net2(grayscale_with, mask)                      # grayscale_with

        # combination of learned information with grayscale_with
        out_wholeimg = grayscale_with * (1 - mask) + out_decoder * mask  

        # Validation of grayscale_with -> original image with
        L1_list_with2with.append(L1_Loss(out_ae,grayscale_with ).item())
        L2_list_with2with.append(L2_Loss(out_ae,grayscale_with ).item())
        Hub_list_with2with.append(Huber_Loss(out_ae,grayscale_with ).item())
        Focal_list_with2with.append(FocalFrequency_Loss(out_ae,grayscale_with ).item())
        lpips_list_with2with.append(iqa_metric_lpips(out_ae,grayscale_with).item())
        psnr_list_with2with.append(iqa_metric_psnr(out_ae, grayscale_with).item())
        ssim_list_with2with.append(iqa_metric_ssim(out_ae, grayscale_with).item())

        # Validation of grayscale_without -> original image with
        L1_list_without2with.append(L1_Loss(out_decoder,grayscale_with ).item())
        L2_list_without2with.append(L2_Loss(out_decoder,grayscale_with ).item())
        Hub_list_without2with.append(Huber_Loss(out_decoder,grayscale_with ).item())
        Focal_list_without2with.append(FocalFrequency_Loss(out_decoder,grayscale_with ).item())
        lpips_list_without2with.append(iqa_metric_lpips(out_decoder,grayscale_with).item())
        psnr_list_without2with.append(iqa_metric_psnr(out_decoder, grayscale_with).item())
        ssim_list_without2with.append(iqa_metric_ssim(out_decoder, grayscale_with).item())

        utils_sample.sample_general_bottleneck(opt, grayscale_with, out_ae, grayscale_without, out_decoder, opt.sample_path, (1 + 1),saving_name=str('validation'))


    logg(f"lpips_list_with2with is : {lpips_list_with2with}")
    logg(f"lpips_list_without2with is : {lpips_list_without2with}")
    logg("---------------------------------------------")

    logg(f"psnr_list_with2with is : {psnr_list_with2with}")
    logg(f"psnr_list_without2with is : {psnr_list_without2with}")
    logg("---------------------------------------------")

    logg(f"ssim_list_with2with is : {ssim_list_with2with}")
    logg(f"ssim_list_without2with is : {ssim_list_without2with}")
    logg("---------------------------------------------")

    logg(f"L1_list_with2with are : {L1_list_with2with}")
    logg(f"L1_list_without2with are : {L1_list_without2with}")
    logg("---------------------------------------------")

    logg(f"L2_list_with2with are : {L2_list_with2with}")
    logg(f"L2_list_without2with are : {L2_list_without2with}")
    logg("---------------------------------------------")

    logg(f"Hub_list_with2with are : {Hub_list_with2with}")
    logg(f"Hub_list_without2with are : {Hub_list_without2with}")
    logg("---------------------------------------------")

    logg(f"Focal_list_with2with are : {Focal_list_with2with}")
    logg(f"Focal_list_without2with are : {Focal_list_without2with}")
    logg("---------------------------------------------")

    # AVERAGES with2with
    logg("---------------------------------------------")
    logg(f"lpips_list_with2with : {np.average(lpips_list_with2with):.4f}  \n psnr_list_with2with : {np.average(psnr_list_with2with):.4f} \n ssim_list_with2with : {np.average(ssim_list_with2with):.4f} ")
    logg(f"L1_list_with2with : {np.average(L1_list_with2with):.4f}  \n L2_list_with2with : {np.average(L2_list_with2with):.4f} \n Hub_list_with2with : {np.average(Hub_list_with2with):.4f} \n Focal_list_with2with : {np.average(Focal_list_with2with):.4f} ")

    # AVERAGES without2with
    logg("---------------------------------------------")
    logg(f"lpips_list_without2with : {np.average(lpips_list_without2with):.4f} \n psnr_list_without2with  : {np.average(psnr_list_without2with):.4f} \n ssim_list_without2with : {np.average(ssim_list_without2with):.4f} ")
    logg(f"L1_list_without2with : {np.average(L1_list_without2with):.4f} \n L2_list_without2with  : {np.average(L2_list_without2with):.4f} \n Hub_list_without2with : {np.average(Hub_list_without2with):.4f} \n Focal_list_without2with : {np.average(Focal_list_without2with):.4f}")


def Validate_prototype_4Y_0(opt):
   
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------
    opt.in_channels = 3
    if opt.in_channels == 1:

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((opt.imgsize,opt.imgsize)) #, torchvision.transforms.RandomCrop(opt.imgsize)
                           ])

    else:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((opt.imgsize,opt.imgsize))
                # torchvision.transforms.CenterCrop(opt.imgsize)
                                ])
    opt.in_channels = 1
    opt.mask_channels = 1
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    # Build networks
    encoder_bottle_net = utils_networks.create_encoder(opt)
    decoder_net = utils_networks.create_decoder(opt)
    bottleneck_net = utils_networks.create_bottleneck(opt)
    # ae_net = utils.create_AE(opt)
    ae_net2 = utils_networks.create_AE2(opt)

    # To device
    if opt.multi_gpu == True:
        encoder_bottle_net = nn.DataParallel(encoder_bottle_net)
        decoder_net = nn.DataParallel(decoder_net)
        bottleneck_net = nn.DataParallel(bottleneck_net)
        ae_net2 = nn.DataParallel(ae_net2)
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()
    else:
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()




    
    if opt.validation == 1:
        

        pytorch_total_params = sum(p.numel() for p in bottleneck_net.parameters())
        print(f'bottleneck_net :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in ae_net2.parameters())
        print(f'ae_net2 :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in encoder_bottle_net.parameters())
        print(f'encoder_bottle_net :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in decoder_net.parameters())
        print(f'decoder_net :{pytorch_total_params}')

        # print("#"*40)
        # print(pyiqa.list_models())


        opt.baseroot = opt.test_base
        opt.baseroot_pair = opt.test_base_pair
        # trainset = dataset.InpaintDataset_pairs(opt, transforms=transforms) # InpaintDataset_pairs_without_mask_vadilation
        trainset = dataset.InpaintDataset_pairs_validation(opt, transforms=transforms)
        print('The overall number of images equals to %d' % len(trainset))

        # Define the dataloader
        # dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
        dataloader = DataLoader(trainset, batch_size = 1, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

        for batch_idx, (grayscale_without, grayscale_without2, mask) in enumerate(dataloader):

            # grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()
            grayscale_without2 = grayscale_without2.cuda()
            mask = mask.cuda()

            # inference
            out_enc_net1 = encoder_bottle_net(grayscale_without, mask)  # grayscale_without
            out_bottle =  bottleneck_net(out_enc_net1) 
            out_decoder = ae_net2.decoder(out_bottle)   

            out_enc_net2 = encoder_bottle_net(grayscale_without2, mask)  # grayscale_without
            out_bottle2 =  bottleneck_net(out_enc_net2) 
            out_decoder2 = ae_net2.decoder(out_bottle2)   
            # out_ae = ae_net2(grayscale_without)                      # grayscale_with


            utils_sample.sample_general_bottleneck(opt, grayscale_without, out_decoder, grayscale_without2, out_decoder2, opt.sample_path, (1 + 1),saving_name=str(f'pix2pix_4Y_{batch_idx}'))
