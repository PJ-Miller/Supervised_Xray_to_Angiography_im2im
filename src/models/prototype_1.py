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
import pyiqa
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score 

# # Seed
# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

'''
Basic Idea:
    only basic autoencoder included 
    is idea1 in model_sum_2.ppt
    trains 2 autoencoder and then learns a bottleneck network 
    between the two laten spaces
    2 autoencoder training happens first, then parameter are frozen and then only bottleneck is trained
    Different options in Loss functions 
    (Loss between latent spaces, loss between decoder outputs, loss function between enc - bottle -bottle - decod2 and pair_image)
'''

def Prototype_1(opt):

    # ---------------------------------------------------------------------------------------#
    # ------------------------    Initialize training parameters    ------------------------ #
    # ---------------------------------------------------------------------------------------#

    logg("Initializing pretraining function of prototype 1")
    
    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    logg("There are %d GPUs used" % gpu_num)
    # print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    logg("Batch size is changed to %d" % opt.batch_size)
    logg("Number of workers is changed to %d" % opt.num_workers)

    # print("Batch size is changed to %d" % opt.batch_size)
    # print("Number of workers is changed to %d" % opt.num_workers)
    # Build path folder
    utils_core.check_path(opt.save_path)
    utils_core.check_path(opt.sample_path)

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
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()
    else:
        encoder_bottle_net = encoder_bottle_net.cuda()
        decoder_net = decoder_net.cuda()
        bottleneck_net = bottleneck_net.cuda()
        ae_net2 = ae_net2.cuda()

    # Loss functions
    if opt.loss_function == "L1":
        LossFunction = nn.L1Loss()
    elif opt.loss_function == "L2":
        LossFunction = nn.MSELoss()
    elif opt.loss_function == "Huber":
        LossFunction = nn.HuberLoss()
    elif opt.loss_function == "Focal":
        LossFunction = focal_frequency_loss.FocalFrequencyLoss()

    # print(f"Loss function used : {opt.loss_function}")
    logg(f"Loss function used : {opt.loss_function}")


    # Optimizers
    optimizer_enc = torch.optim.Adam(encoder_bottle_net.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_dec = torch.optim.Adam(decoder_net.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_bottle = torch.optim.Adam(bottleneck_net.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_ae2 = torch.optim.Adam(ae_net2.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt, name):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = f'Model_version_{opt.saving_name}_{name}_{opt.loss_function}_{opt.gan_loss_function}_epoch{epoch+20}_batchsize{opt.batch_size}.pth'
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                logg(f'The trained model is successfully saved at epoch {epoch} as {model_name}')

        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                logg(f'The trained model is successfully saved at epoch {epoch} as {model_name}')
                # print(f'The trained model is successfully saved at epoch {epoch} as {model_name}')

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    if opt.in_channels == 1:
        # transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(1), torchvision.transforms.CenterCrop(opt.imgsize)
        #                     ]) # ,torchvision.transforms.Resize(opt.imgsize+50)
#     if opt.in_channels == 1:
        # transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(1), torchvision.transforms.Resize(opt.imgsize+100) ,torchvision.transforms.RandomCrop(opt.imgsize)
        #                     ])
        # transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((750,1125)), torchvision.transforms.RandomCrop(opt.imgsize)
        #                    ])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((opt.imgsize,opt.imgsize)) #, torchvision.transforms.RandomCrop(opt.imgsize)
                           ])
        # transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(1) ,torchvision.transforms.Resize((int(opt.imgsize*1.5), opt.imgsize)) , torchvision.transforms.RandomCrop(opt.imgsize)
        #                    ])
    else:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(opt.imgsize)
                                ])
# , torchvision.transforms.ToTensor()
    # transforms = torch.nn.Sequential(
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.CenterCrop(10),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # )   

    # Define the dataset
    # trainset = dataset.InpaintDataset(opt, transforms=transforms)
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

            # grayscale pair  pipeline
            grayscale_with = grayscale_with.cuda()
            optimizer_ae2.zero_grad()
            out_ae_pair = ae_net2(grayscale_with, mask)    
            out_wholeimg_pair = grayscale_with * (1 - mask) + out_ae_pair * mask  
            MaskL1Loss_pair = LossFunction(out_wholeimg_pair, grayscale_with)
            MaskL1Loss_pair.backward()
            optimizer_ae2.step()

            ## grayscale pipeline
            grayscale_without = grayscale_without.cuda()
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            out_enc_net1 = encoder_bottle_net(grayscale_without, mask)    
            out_ae = decoder_net(out_enc_net1)   
            out_wholeimg = grayscale_without * (1 - mask) + out_ae * mask
            MaskL1Loss = LossFunction(out_wholeimg, grayscale_without)
            MaskL1Loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            
            # save_image(grayscale_with[0], 'grayscale_with.png')
            # save_image(grayscale_without[0], 'grayscale_without.png')

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if epoch % opt.log_interval == 0:
                logg(f"[ Epoch {(epoch + 1)}/{opt.epochs} ] [ Batch {batch_idx}/{len(dataloader)} ]   \t[ Loss function : {opt.loss_function} Loss_with: {MaskL1Loss_pair.item():.5f}  Loss_without: {MaskL1Loss.item():.5f} ]\
                        time_left: {time_left}")


        # Learning rate decrease
        adjust_learning_rate(optimizer_enc, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_dec, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_ae2, (epoch + 1), opt, opt.lr_g)

        # Save the model

        save_model(encoder_bottle_net, (epoch + 1), opt, '_Enc_without_')
        save_model(decoder_net, (epoch + 1), opt, '_Dec_without_')
        
        # save_model(ae_net, (epoch + 1), opt, 'autoen1')
        save_model(ae_net2, (epoch + 1), opt, '_AE_with_')

        if epoch % opt.checkpoint_image== opt.checkpoint_image-1:
            # utils.sample(opt, grayscale, mask, out_wholeimg1, out1, opt.sample_path, (epoch + 1))
            # grayscale_with/without : input img, mask : mask, out_wholeimg/_pair : combination of output+unmasked imag , out_ae/_pair : only output
            utils_sample.sample_general(opt, grayscale_with, mask, out_wholeimg_pair, out_ae_pair, opt.sample_path, (epoch + 1),saving_name=str(opt.saving_name+'_with'))
            utils_sample.sample_general(opt, grayscale_without, mask, out_wholeimg, out_ae, opt.sample_path, (epoch + 1),saving_name=opt.saving_name+'_without')

    # quit()
    for param in ae_net2.parameters():
        param.requires_grad = False

    for param in encoder_bottle_net.parameters():
        param.requires_grad = False

    for param in decoder_net.parameters():
        param.requires_grad = False

    logg("bottleneck training!")
    # Initialize start time
    prev_time = time.time()

    for epoch in range(opt.epochs): 
        for batch_idx, (grayscale_without, grayscale_with, mask) in enumerate(dataloader): 
            
            # print(grayscale.shape, mask.shape)

            grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()                                    # out: [B, 1, 256, 256]
            mask = mask.cuda()                                              # out: [B, 1, 256, 256]
            
            # forward propagation
            optimizer_bottle.zero_grad()
            
            # split Encoder Decoder input : without , output : with
            out_enc_net1 = encoder_bottle_net(grayscale_without, mask)   # grayscale_pair
            out_bottle =  bottleneck_net(out_enc_net1) 
            out_decoder = ae_net2.decoder(out_bottle)   

            # output of autoencoder 
            out_ae = ae_net2(grayscale_with, mask)
            # out_wholeimg1 = grayscale * (1 - mask) + out1 * mask              # in range [0, 1]
            out_wholeimg = grayscale_with * (1 - mask) + out_decoder * mask  


            # Mask L1 Loss
            MaskL1Loss = LossFunction(out_decoder, out_ae)

            # loss1.backward()
            MaskL1Loss.backward()

            optimizer_bottle.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if epoch % opt.log_interval == 0:
                logg(f"[Epoch {(epoch + 1)}/{opt.epochs}]\t[Batch {batch_idx}/{len(dataloader)}]   \t[Loss function: {opt.loss_function}  Loss: {MaskL1Loss.item():.5f} ]\ttime_left: {time_left}",logs= (epoch+1)%opt.log_interval==0)

        # Learning rate decrease
        # adjust_learning_rate(optimizer_enc, (epoch + 1), opt, opt.lr_g)
        # adjust_learning_rate(optimizer_dec, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_bottle, (epoch + 1), opt, opt.lr_g)

        # Save the model
        save_model(bottleneck_net, (epoch + 1), opt, '_bottleneck_')
        # save_model(decoder_net, (epoch + 1), opt, 'dec')
        # # save_model(ae_net, (epoch + 1), opt, 'autoen1')
        # save_model(ae_net2, (epoch + 1), opt, 'autoen2')
        if epoch % opt.checkpoint_image== opt.checkpoint_image-1:
            # utils.sample(opt, grayscale, mask, out_wholeimg1, out1, opt.sample_path, (epoch + 1))
            # utils_sample.sample_general_bottleneck(opt, grayscale_without, out_wholeimg, out_ae, out_decoder, opt.sample_path, (epoch + 1),saving_name=str(opt.saving_name+'_bottleneck'))
            utils_sample.sample_general_bottleneck(opt, grayscale_without, grayscale_with, out_ae, out_decoder, opt.sample_path, (epoch + 1),saving_name=str(opt.saving_name+'_bottleneck'))

            # utils.sample(opt, grayscale, mask, out_wholeimg, out_ae, opt.sample_path, (epoch + 2))


    if opt.validation == 1:
        
        # print all model parameters
        # bottleneck network : bottleneck_net
        # autoencoder with Vessel information : ae_net2
        # encoder without Vessel information : encoder_bottle_net
        # decoder without Vessel information : decoder_net
        pytorch_total_params = sum(p.numel() for p in bottleneck_net.parameters())
        logg(f'bottleneck_net :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in ae_net2.parameters())
        logg(f'ae_net2 :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in encoder_bottle_net.parameters())
        logg(f'encoder_bottle_net :{pytorch_total_params}')

        pytorch_total_params = sum(p.numel() for p in decoder_net.parameters())
        logg(f'decoder_net :{pytorch_total_params}')

        logg("#"*40)
        logg(pyiqa.list_models())


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

        norm_MI_with2with = []
        norm_MI_without2with = []

        adj_MI_with2with = []
        adj_MI_without2with = []

        MI_with2with = []
        MI_without2with = [] 

        cor_with2with = []
        cor_without2with = [] 
        # run throught test set

        opt.baseroot = opt.test_base
        opt.baseroot_pair = opt.test_base_pair
        trainset = dataset.InpaintDataset_pairs(opt, transforms=transforms)

        logg('The overall number of images equals to %d' % len(trainset))

        # Define the dataloader
        # dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
        dataloader = DataLoader(trainset, batch_size = 1, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

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

            output_ae_int = (out_ae.detach().to('cpu').ravel()*255).to(torch.int16)
            output_decoder_int = (out_decoder.detach().to('cpu').ravel()*255).to(torch.int16)
            original_with_int = (grayscale_with.detach().to('cpu').ravel()*255).to(torch.int16)

            # Validation of grayscale_with -> original image with
            L1_list_with2with.append(L1_Loss(out_ae,grayscale_with ).item())
            L2_list_with2with.append(L2_Loss(out_ae,grayscale_with ).item())
            Hub_list_with2with.append(Huber_Loss(out_ae,grayscale_with ).item())
            Focal_list_with2with.append(FocalFrequency_Loss(out_ae,grayscale_with ).item())
            lpips_list_with2with.append(iqa_metric_lpips(out_ae,grayscale_with).detach().to('cpu').numpy())
            psnr_list_with2with.append(iqa_metric_psnr(out_ae, grayscale_with).detach().to('cpu').numpy())
            ssim_list_with2with.append(iqa_metric_ssim(out_ae, grayscale_with).detach().to('cpu').numpy())
            norm_MI_with2with.append(normalized_mutual_info_score(output_ae_int, original_with_int))
            adj_MI_with2with.append(adjusted_mutual_info_score(output_ae_int, original_with_int))
            MI_with2with.append(mutual_info_score(output_ae_int, original_with_int))
            cor_with2with.append(np.corrcoef(output_ae_int, original_with_int)[0][1])

            # Validation of grayscale_without -> original image with
            L1_list_without2with.append(L1_Loss(out_decoder,grayscale_with ).item())
            L2_list_without2with.append(L2_Loss(out_decoder,grayscale_with ).item())
            Hub_list_without2with.append(Huber_Loss(out_decoder,grayscale_with ).item())
            Focal_list_without2with.append(FocalFrequency_Loss(out_decoder,grayscale_with ).item())
            lpips_list_without2with.append(iqa_metric_lpips(out_decoder,grayscale_with).detach().to('cpu').numpy())
            psnr_list_without2with.append(iqa_metric_psnr(out_decoder, grayscale_with).detach().to('cpu').numpy())
            ssim_list_without2with.append(iqa_metric_ssim(out_decoder, grayscale_with).detach().to('cpu').numpy())
            norm_MI_without2with.append(normalized_mutual_info_score(output_decoder_int, original_with_int))
            adj_MI_without2with.append(adjusted_mutual_info_score(output_decoder_int, original_with_int))
            MI_without2with.append(mutual_info_score(output_decoder_int, original_with_int))
            cor_without2with.append(np.corrcoef(output_decoder_int, original_with_int)[0][1])

            utils_sample.sample_general_bottleneck(opt, grayscale_with, out_ae, grayscale_without, out_decoder, opt.sample_path, (1 + 1),saving_name=str(f'validation_{batch_idx}'))


        # lpips_list_with2with = [item for sublist in lpips_list_with2with for item in sublist]
        # lpips_list_without2with = [item for sublist in lpips_list_without2with for item in sublist]
        # psnr_list_with2with = [item for sublist in psnr_list_with2with for item in sublist]
        # psnr_list_without2with = [item for sublist in psnr_list_without2with for item in sublist]
        # ssim_list_with2with = [item for sublist in ssim_list_with2with for item in sublist]
        # ssim_list_without2with = [item for sublist in ssim_list_without2with for item in sublist]


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

        logg(f"norm_MI_with2with are : {norm_MI_with2with}")
        logg(f"norm_MI_without2with are : {norm_MI_without2with}")
        logg("---------------------------------------------")

        logg(f"adj_MI_with2with are : {adj_MI_with2with}")
        logg(f"adj_MI_without2with are : {adj_MI_without2with}")
        logg("---------------------------------------------")

        logg(f"MI_with2with are : {MI_with2with}")
        logg(f"MI_without2with are : {MI_without2with}")
        logg("---------------------------------------------")


        logg(f"cor_with2with are : {cor_with2with}")
        logg(f"cor_without2with are : {cor_without2with}")
        logg("---------------------------------------------")


        # AVERAGES with2with
        logg("---------------------------------------------")
        logg(f"lpips_list_with2with : {np.average(lpips_list_with2with):.6f}  \n psnr_list_with2with : {np.average(psnr_list_with2with):.6f} \n ssim_list_with2with : {np.average(ssim_list_with2with):.6f} ")
        logg(f"L1_list_with2with : {np.average(L1_list_with2with):.6f}  \n L2_list_with2with : {np.average(L2_list_with2with):.6f} \n Hub_list_with2with : {np.average(Hub_list_with2with):.6f} \n Focal_list_with2with : {np.average(Focal_list_with2with):.6f} ")

        logg(f"norm_MI_with2with : {np.average(norm_MI_with2with):.6f}  \n adj_MI_with2with : {np.average(adj_MI_with2with):.6f} \n MI_with2with : {np.average(MI_with2with):.6f} \n cor_with2with : {np.average(cor_with2with):.6f}")

        # AVERAGES without2with
        logg("---------------------------------------------")
        logg(f"lpips_list_without2with : {np.average(lpips_list_without2with):.6f} \n psnr_list_without2with  : {np.average(psnr_list_without2with):.6f} \n ssim_list_without2with : {np.average(ssim_list_without2with):.6f} ")
        logg(f"L1_list_without2with : {np.average(L1_list_without2with):.6f} \n L2_list_without2with  : {np.average(L2_list_without2with):.6f} \n Hub_list_without2with : {np.average(Hub_list_without2with):.6f} \n Focal_list_without2with : {np.average(Focal_list_without2with):.6f}")

        logg(f"norm_MI_without2with : {np.average(norm_MI_without2with):.6f}  \n adj_MI_without2with : {np.average(adj_MI_without2with):.6f} \n MI_without2with : {np.average(MI_without2with):.6f} \n cor_without2with : {np.average(cor_without2with):.6f}")
