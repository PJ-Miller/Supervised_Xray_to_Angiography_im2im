################ imports ################
import torch
import numpy as np
import random
from ..utils.utils_core import logg
import torch.backends.cudnn as cudnn
from ..utils import utils_core, utils_networks, utils_sample
import torch.nn as nn
from ..metrics import focal_frequency_loss
import os
import torchvision
from ..dataloaders import dataset
import time
from torch.utils.data import DataLoader
import datetime

'''
    most basic idea came last
    is idea3 in model_sum_2.ppt
    trains 1 autoencoder  
    will try to learn mapping from input without nervers to output with nerves
'''

def Prototype_3(opt):

    # ---------------------------------------------------------------------------------------#
    # ------------------------    Initialize training parameters    ------------------------ #
    # ---------------------------------------------------------------------------------------#

    logg("Initializing pretraining function option 0")

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    logg("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    logg("Batch size is changed to %d" % opt.batch_size)
    logg("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils_core.check_path(opt.save_path)
    utils_core.check_path(opt.sample_path)


    # Build networks
    ae_net = utils_networks.create_AE2(opt)
    # unet = utils.create_Unet()

    # To device
    if opt.multi_gpu == True:
        ae_net = nn.DataParallel(ae_net)
        ae_net = ae_net.cuda()
    else:
        ae_net = ae_net.cuda()

    # Loss functions
    if opt.loss_function == "L1":
        LossFunction = nn.L1Loss()
    elif opt.loss_function == "L2":
        LossFunction = nn.MSELoss()
    elif opt.loss_function == "Huber":
        LossFunction = nn.HuberLoss()
    elif opt.loss_function == "Focal":
        LossFunction = focal_frequency_loss.FocalFrequencyLoss()

    logg(f"Loss function used : {opt.loss_function}")
        
    # Optimizers
    optimizer_ae = torch.optim.Adam(ae_net.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt, name):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = f'Model_{name}_{opt.loss_function}_{opt.gan_loss_function}_epoch{epoch+20}_batchsize{opt.batch_size}.pth'
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                logg('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                logg('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(1),
                        # torchvision.transforms.CenterCrop(opt.imgsize)
                        ])
                        
    # torchvision.transforms.ToTensor()
    # transforms = torchvision.transforms.Compose([
    #     # torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Grayscale(1),
    #     torchvision.transforms.CenterCrop(opt.imgsize),
    #     # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])   

    # Define the dataset
    trainset = dataset.InpaintDataset_pairs(opt, transforms=transforms)

    logg('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale_without, grayscale_with, mask) in enumerate(dataloader):
            

            mask = mask.cuda()                                              # out: [B, 1, 256, 256]
            grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()
            
            # save_image(grayscale_with[0], 'grayscale_with.png')
            # save_image(grayscale_without[0], 'grayscale_without.png')

            optimizer_ae.zero_grad()

            out_with = ae_net(grayscale_without, mask)  

            out_wholeimg_with = grayscale_with * (1 - mask) + out_with * mask  
            MaskLoss = LossFunction(out_wholeimg_with, grayscale_with)
            MaskLoss.backward()

            optimizer_ae.step()


            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            logg(f"[Epoch {(epoch + 1)}/{opt.epochs}] \t [Batch {batch_idx}/{len(dataloader)}] [Loss function: {opt.loss_function}  MaskLoss: {MaskLoss.item():.5f} ] \t time_left: {time_left}")
            

        # Learning rate decrease
        adjust_learning_rate(optimizer_ae, (epoch + 1), opt, opt.lr_g)

        # Save the model
        save_model(ae_net, (epoch + 1), opt, 'autoencder_idea3_')

        if epoch % opt.checkpoint_image== opt.checkpoint_image-1:
            utils_sample.sample_general(opt, grayscale_without, mask, out_wholeimg_with, out_with, opt.sample_path, (epoch + 1),saving_name=opt.saving_name+'_new')
