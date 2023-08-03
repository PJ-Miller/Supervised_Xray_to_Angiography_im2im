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
# import p6yiqa
# import time
from torch.utils.data import DataLoader
# import datetime
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score 





'''
Fast script that just test data without ground truth, runs them through models and saves the results

'''

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

def Validate_prototype_4Y(opt):
   
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    # testset = dataset.InpaintDataset(opt)
    # print('The overall number of images equals to %d' % len(testset))

    # # Define the dataloader
    # dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # if opt.in_channels == 1:
    #     transforms = torchvision.transforms.Compose([
    #         torchvision.transforms.Grayscale(1), torchvision.transforms.CenterCrop(opt.imgsize)
    #                         ])
    # else:
    #     transforms = torchvision.transforms.Compose([
    #             torchvision.transforms.CenterCrop(opt.imgsize)
    #                             ])
    # if opt.in_channels == 1:

    #     transforms = torchvision.transforms.Compose([
    #         torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((opt.imgsize,opt.imgsize)) #, torchvision.transforms.RandomCrop(opt.imgsize)
    #                        ])

    # else:
    #     transforms = torchvision.transforms.Compose([
    #             torchvision.transforms.CenterCrop(opt.imgsize)
    #                             ])

    opt.in_channels = 3
    # opt.in_channels = 3

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
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    # Build networks
    encoder_bottle_net = utils_networks.create_encoder_basic(opt)
    decoder_net = utils_networks.create_decoder_basic(opt)
    bottleneck_net = utils_networks.create_bottleneck_basic(opt)
    # ae_net = utils.create_AE(opt)
    ae_net2 = utils_networks.create_AEbasic(opt)

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
        
        # print all model parameters
        # bottleneck network : bottleneck_net
        # autoencoder with Vessel information : ae_net2
        # encoder without Vessel information : encoder_bottle_net
        # decoder without Vessel information : decoder_net
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


        # print("----------- LIBRARY NEW METRICS -------------")
        # print("---------------------------------------------")

        # iqa_metric_lpips = pyiqa.create_metric('lpips', device=torch.device('cuda'))
        # print(f"lower_better is : {iqa_metric_lpips.lower_better}")

        # # iqa_metric_fid = pyiqa.create_metric('fid', device=torch.device('cuda'))
        # # logging.info(f"lower_better is : {iqa_metric_fid.lower_better}")

        # iqa_metric_psnr = pyiqa.create_metric('psnr', device=torch.device('cuda'))
        # print(f"lower_better is : {iqa_metric_psnr.lower_better}")

        # iqa_metric_ssim = pyiqa.create_metric('ssim', device=torch.device('cuda'))
        # print(f"lower_better is : {iqa_metric_ssim.lower_better}")
        # iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='gray')

        opt.baseroot = opt.test_base
        opt.baseroot_pair = opt.test_base_pair
        trainset = dataset.InpaintDataset_pairs_validation(opt, transforms=transforms)

        print('The overall number of images equals to %d' % len(trainset))

        # Define the dataloader
        # dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
        dataloader = DataLoader(trainset, batch_size = 1, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

        for batch_idx, (grayscale_without, grayscale_without2, _) in enumerate(dataloader):

            # grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()
            grayscale_without2 = grayscale_without2.cuda()
            # mask = mask.cuda()
            
            # inference
            out_enc_net1 = encoder_bottle_net(grayscale_without)  # grayscale_without
            out_bottle =  bottleneck_net(out_enc_net1) 
            out_decoder = ae_net2.decoder(out_bottle)   

            out_ae = decoder_net(out_enc_net1)

            out_enc_net2 = encoder_bottle_net(grayscale_without2)  # grayscale_without
            out_bottle2 =  bottleneck_net(out_enc_net2) 
            out_decoder2 = ae_net2.decoder(out_bottle2)   

            out_ae2 = decoder_net(out_enc_net2)
            # out_ae = ae_net2(grayscale_without)                      # grayscale_with


            utils_sample.sample_general_bottleneck(opt, grayscale_without, out_decoder, grayscale_without2, out_decoder2, opt.sample_path, (1 + 1),saving_name=str(f'pix2pix_4Y_{batch_idx}'))

'''
python .\run_validation_prototype_4Y.py --finetune_path_decoder D:\\thesis_pix2pix\\models\\Model_version_prototype_1__Dec_without__L1_L2_epoch60_batchsize1.pth --finetune_path_encoder D:\\thesis_pix2pix\\models\\Model_version_prototype_1__Enc_without__L1_L2_epoch60_batchsize1.pth --finetune_path_autoencoder D:\\thesis_pix2pix\\models\\Model_version_prototype_1__AE_with__L1_L2_epoch60_batchsize1.pth --finetune_path_bottleneck D:\\thesis_pix2pix\\models\\Model_version_prototype_1__bottleneck__L1_L2_epoch40_batchsize1.pth


--finetune_path_encoder D:\\thesis_pix2pix\\models\\Model_version_TAKIS__Enc_without__L1_L2_epoch70_batchsize1.pth
--finetune_path_decoder D:\\thesis_pix2pix\\models\\Model_version_TAKIS__Dec_without__L1_L2_epoch70_batchsize1.pth
--finetune_path_autoencoder D:\\thesis_pix2pix\\models\\Model_version_TAKIS__AE_with__L1_L2_epoch70_batchsize1.pth
--finetune_path_bottleneck D:\\thesis_pix2pix\\models\\Model_version_TAKIS__bottleneck__L1_L2_epoch70_batchsize1.pth
'''
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
