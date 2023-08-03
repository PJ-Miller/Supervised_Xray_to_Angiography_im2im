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
    # basic autoencoder training images with nerves 
    # and encoder plus bottleneck combined with decoder from images with nerves, trained with reconstruction and adversarial loss 
    # is idea2 in model_sum_2.ppt
    # autoencoder with nervers is trained first , then froze and discriminator is trained
    # and then encoder+bottleneck is trained 
'''
def Prototype_2(opt):


    # ---------------------------------------------------------------------------------------#
    # ------------------------    Initialize training parameters    ------------------------ #
    # ---------------------------------------------------------------------------------------#
    print("Initializing pretraining function")

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
    

    # Build the networks
    autoencoder_with = utils_networks.create_AE2(opt)
    encBot_without = utils_networks.create_encoder(opt)
    discriminator = utils_networks.create_discriminator224(opt)

    # To device
    if opt.multi_gpu == True:
        autoencoder_with = nn.DataParallel(autoencoder_with)
        encBot_without = nn.DataParallel(encBot_without)
        discriminator = nn.DataParallel(discriminator)
        autoencoder_with = autoencoder_with.cuda()
        encBot_without = encBot_without.cuda()
        discriminator = discriminator.cuda()
    else:
        autoencoder_with = autoencoder_with.cuda()
        encBot_without = encBot_without.cuda()
        discriminator = discriminator.cuda()


    # Loss functions
    if opt.loss_function == "L1":
        LossFunction = nn.L1Loss()
    elif opt.loss_function == "L2":
        LossFunction = nn.MSELoss()
    elif opt.loss_function == "Huber":
        LossFunction = nn.HuberLoss()
    elif opt.loss_function == "Focal":
        LossFunction = focal_frequency_loss.FocalFrequencyLoss()

    if opt.gan_loss_function == "L1":
        GanLossFunction = nn.L1Loss()
    elif opt.gan_loss_function == "L2":
        GanLossFunction = nn.MSELoss()
    elif opt.gan_loss_function == "Huber":
        GanLossFunction = nn.HuberLoss()
    elif opt.gan_loss_function == "Focal":
        GanLossFunction = focal_frequency_loss.FocalFrequencyLoss()

    logg(f"Loss function used : {opt.loss_function}")
    logg(f"GAN Loss function used : {opt.gan_loss_function}")

    # Optimizers
    optimizer_autoencoder_with = torch.optim.Adam(autoencoder_with.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_encBot_without = torch.optim.Adam(encBot_without.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    # def save_model(net, epoch, opt):
    #     """Save the model at "checkpoint_interval" and its multiple"""
    #     # model_name = 'GrayInpainting_GAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
    #     model_name = f'Model_{opt.model}_{opt.loss_function}_{opt.gan_loss_function}_{opt.mask_type}_epoch{epoch+20}_batchsize{opt.batch_size}.pth'

    #     model_path = os.path.join(opt.save_path, model_name)
    #     if opt.multi_gpu == True:
    #         if epoch % opt.checkpoint_interval == 0:
    #             torch.save(net.module.state_dict(), model_path)
    #             print('The trained model is successfully saved at epoch %d' % (epoch))
    #     else:
    #         if epoch % opt.checkpoint_interval == 0:
    #             torch.save(net.state_dict(), model_path)
    #             print('The trained model is successfully saved at epoch %d' % (epoch))
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
    if opt.in_channels == 1:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((opt.imgsize,opt.imgsize)) #, torchvision.transforms.RandomCrop(opt.imgsize)
                           ])
    else:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(opt.imgsize)
                                ])               
        
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

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale_without, grayscale_with, mask) in enumerate(dataloader):
            
            mask = mask.cuda()                                             
            grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()

            # save_image(grayscale_with[0], 'grayscale_with.png')
            # save_image(grayscale_without[0], 'grayscale_without.png')
            # LSGAN vectors # CHANGE THIS
            valid = Tensor(np.ones((grayscale_with.shape[0], 1, 8, 8)))
            fake = Tensor(np.zeros((grayscale_without.shape[0], 1, 8, 8)))

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_discriminator.zero_grad()

            # forward propagation
            out_with = autoencoder_with(grayscale_with, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg_with = grayscale_with * (1 - mask) + out_with * mask              # in range [0, 1]

            out_without = autoencoder_with.decoder(encBot_without(grayscale_without, mask))                        # out: [B, 1, 256, 256]
            out_wholeimg_without = grayscale_with * (1 - mask) + out_without * mask                             # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(out_wholeimg_without.detach(), mask)


            # True samples
            true_scalar = discriminator(out_wholeimg_with.detach(), mask)
            # Overall Loss and optimize
            loss_fake = GanLossFunction(fake_scalar, fake)
            loss_true = GanLossFunction(true_scalar, valid)
            # Overall Loss and optimize
            loss_Discriminator = 0.5 * (loss_fake + loss_true)
            loss_Discriminator.backward()

            optimizer_discriminator.step()

            # ----------------------------------------
            #         Train autoencoders with
            # ----------------------------------------
            optimizer_autoencoder_with.zero_grad()

            # forward propagation with
            out_with = autoencoder_with(grayscale_with, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg_with = grayscale_with * (1 - mask) + out_with * mask              # in range [0, 1]



            # Mask Loss with
            MaskLoss_with = LossFunction(out_wholeimg_with, grayscale_with)
            loss_with = opt.lambda_loss_function * MaskLoss_with
            loss_with.backward()
            optimizer_autoencoder_with.step()

            # ----------------------------------------
            #         Train autoencoders without
            # ----------------------------------------

            optimizer_encBot_without.zero_grad()

            # forward propagation without
            out_without = autoencoder_with.decoder(encBot_without(grayscale_without, mask))                        # out: [B, 1, 256, 256]
            out_wholeimg_without = grayscale_with * (1 - mask) + out_without * mask                             # in range [0, 1]

            # Mask Loss without
            MaskLoss_without = LossFunction(out_wholeimg_without,grayscale_without)


            # ----------------------------------------
            #                   GAN Loss
            # ----------------------------------------
            fake_scalar = discriminator(out_wholeimg_without, mask)
            MaskGAN_Loss = GanLossFunction(fake_scalar, valid)


            # # ----------------------------------------
            # #              Perceptual Loss
            # # ----------------------------------------

            # # Get the deep semantic feature maps, and compute Perceptual Loss
            # out_3c = torch.cat((out_wholeimg, out_wholeimg, out_wholeimg), 1)
            # grayscale_3c = torch.cat((grayscale, grayscale, grayscale), 1)
            # out_featuremaps = perceptualnet(out_3c)
            # gt_featuremaps = perceptualnet(grayscale_3c)
            # PerceptualLoss = LossFunction(out_featuremaps, gt_featuremaps)


            # Compute losses
            loss_without = opt.lambda_loss_function * MaskLoss_without + opt.lambda_gan * MaskGAN_Loss
            # loss = opt.lambda_loss_function * MaskL1Loss + opt.lambda_perceptual * PerceptualLoss + opt.lambda_gan * MaskGAN_Loss

            loss_without.backward()
            optimizer_encBot_without.step()
            

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            logg(f"[Epoch {(epoch + 1)}/{opt.epochs}] [Batch {batch_idx}/{len(dataloader)}] \t [Loss function: {opt.loss_function} Loss_without: {MaskLoss_without.item():.5f}  GanLoss: {MaskGAN_Loss.item():.5f}  Combined_loss : {loss_without.item():.5f} ] time_left: {time_left}")

        # Learning rate decrease
        adjust_learning_rate(optimizer_encBot_without, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_encBot_without, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_discriminator, (epoch + 1), opt, opt.lr_d)

        # Save the model
        save_model(autoencoder_with, (epoch + 1), opt, 'autoencoder_with_')
        save_model(encBot_without, (epoch + 1), opt, 'encBot_without_')
        save_model(discriminator, (epoch + 1), opt, 'discriminator_')

        if epoch % opt.checkpoint_image== opt.checkpoint_image-1:
            # utils.sample(opt, grayscale, mask, out_wholeimg1, out1, opt.sample_path, (epoch + 1))
            utils_sample.sample_general(opt, grayscale_with, mask, out_wholeimg_with, out_with, opt.sample_path, (epoch + 1),saving_name=str(opt.saving_name+'_autoencoder_with'))
            utils_sample.sample_general(opt, grayscale_without, mask, out_wholeimg_without, out_without, opt.sample_path, (epoch + 1),saving_name=str(opt.saving_name+'_optimizer_encBot_without'))
