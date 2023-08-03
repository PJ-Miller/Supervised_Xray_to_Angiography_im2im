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

def Basic_autoencoder_pix2pix(opt):

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
    ae_net = utils_networks.create_AEbasic(opt)
    # unet = utils.create_Unet()

    discriminator384 = utils_networks.create_discriminator384(opt) # create_discriminator256

    # input384 = torch.rand(1,1,384,384)

    # a = discriminator384(input384)
    # print(a.shape)

    perceptualnet = utils_networks.create_perceptualnet(opt)

    gen_total_params = sum(p.numel() for p in ae_net.parameters())
    disc_total_params = sum(p.numel() for p in discriminator384.parameters())
    per_total_params = sum(p.numel() for p in perceptualnet.parameters())

    print("generator : ", gen_total_params)
    print("discriminator : ", disc_total_params)
    print("perceptualnet : ", per_total_params)

    # To device
    if opt.multi_gpu == True:
        ae_net = nn.DataParallel(ae_net)
        discriminator384 = nn.DataParallel(discriminator384)
        perceptualnet = nn.DataParallel(perceptualnet)
        ae_net = ae_net.cuda()
        discriminator384 = discriminator384.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        ae_net = ae_net.cuda()
        discriminator384 = discriminator384.cuda()
        perceptualnet = perceptualnet.cuda()

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

    # print(f"Loss function used : {opt.loss_function}")
    logg(f"Loss function used : {opt.loss_function}")
    print(f"GAN Loss function used : {opt.gan_loss_function}")


    # Optimizers
    optimizer_ae = torch.optim.Adam(ae_net.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator384.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

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
    # trainset = dataset.InpaintDataset_pairs(opt, transforms=transforms)
    trainset = dataset.InpaintDataset_pairs_without_mask(opt, transforms=transforms)

    logg('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # binary_map = []
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    Rec_loss_values = []
    Per_loss_values = []
    GAN_loss_values = []
    Overall_loss_values = []

    # Training loop
    for epoch in range(opt.epochs):
        # for batch_idx, (grayscale_without, grayscale_with, mask) in enumerate(dataloader):
        running_loss_rec = 0.0
        running_loss_per = 0.0
        running_loss_gan = 0.0
        running_loss_over = 0.0

        for batch_idx, (grayscale_without, grayscale_with) in enumerate(dataloader): 
            # mask2 = torch.zeros(mask.shape)
            # mask = mask2.cuda()  
            # mask = mask.cuda()                                              # out: [B, 1, 256, 256]
            # binary_map.append(mask.mean(dtype=float).float())
            grayscale_with = grayscale_with.cuda()
            grayscale_without = grayscale_without.cuda()

            # LSGAN vectors
            valid = Tensor(np.ones((grayscale_with.shape[0], 1, 12, 12)))
            fake = Tensor(np.zeros((grayscale_with.shape[0], 1, 12, 12)))
            # print(valid.shape, fake.shape)
            # print(grayscale_without.shape, grayscale_without[0].shape)
            # save_image(grayscale_with[0], 'grayscale_with.png')
            # save_image(grayscale_without[0], 'grayscale_without.png')

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_d.zero_grad()

            out_with = ae_net(grayscale_without)  

            # Fake samples
            fake_scalar = discriminator384(out_with.detach())
            # True samples
            true_scalar = discriminator384(grayscale_with)
            # Overall Loss and optimize
            loss_fake = GanLossFunction(fake_scalar, fake)
            loss_true = GanLossFunction(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()

            # ----------------------------------------
            #             Train Generator
            # ----------------------------------------

            optimizer_ae.zero_grad()

            out_with = ae_net(grayscale_without)  

            # GAN Loss
            fake_scalar = discriminator384(out_with)
            MaskGAN_Loss = GanLossFunction(fake_scalar, valid)

            if opt.model ==0 :
                out_wholeimg_with = grayscale_with * (1 - mask) + out_with * mask 
                out_3c = torch.cat((out_wholeimg_with, out_wholeimg_with, out_wholeimg_with), 1)
                grayscale_3c = torch.cat((grayscale_with, grayscale_with, grayscale_with), 1)

                MaskLoss = LossFunction(out_wholeimg_with, grayscale_with)
            elif opt.model == 1:
                out_3c = torch.cat((out_with, out_with, out_with), 1)
                grayscale_3c = torch.cat((grayscale_with, grayscale_with, grayscale_with), 1)

                MaskLoss = LossFunction(out_with, grayscale_with)

            # # GAN Loss
            # fake_scalar = discriminator384(out_with)
            # MaskGAN_Loss = GanLossFunction(fake_scalar, valid)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            # out_3c = torch.cat((out_wholeimg, out_wholeimg, out_wholeimg), 1)
            # grayscale_3c = torch.cat((grayscale, grayscale, grayscale), 1)
            out_featuremaps = perceptualnet(out_3c)
            gt_featuremaps = perceptualnet(grayscale_3c)
            PerceptualLoss = LossFunction(out_featuremaps, gt_featuremaps)

            # Compute losses
            loss = opt.lambda_loss_function * MaskLoss + opt.lambda_perceptual * PerceptualLoss + opt.lambda_gan * MaskGAN_Loss


            loss.backward()

            optimizer_ae.step()


            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            # time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            # prev_time = time.time()

            running_loss_rec =+ MaskLoss.item() * grayscale_without.size(0)
            running_loss_per =+ PerceptualLoss.item() * grayscale_without.size(0)
            running_loss_gan =+ MaskGAN_Loss.item() * grayscale_without.size(0) 
            running_loss_over =+ loss.item() * grayscale_without.size(0) 

            # if epoch % opt.log_interval == 0:
            #     logg(f"[ Epoch {(epoch + 1)}/{opt.epochs} ] [ Batch {batch_idx}/{len(dataloader)} ]   \t[ Pixel_loss : {opt.loss_function} : {MaskLoss.item():.5f} ]\
            #            [ GAN_loss : {opt.gan_loss_function} : {MaskGAN_Loss.item():.5f} ]  \t[ overall : {loss.item():.5f} ] time_left: {time_left}")

        Rec_loss_values.append(running_loss_rec / len(trainset))
        Per_loss_values.append(running_loss_per / len(trainset))
        GAN_loss_values.append(running_loss_gan / len(trainset))
        Overall_loss_values.append(running_loss_over / len(trainset))
        
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
        prev_time = time.time()

        logg(f"[ Epoch {(epoch + 1)}/{opt.epochs} ]   \t[ Pixel_loss : {opt.loss_function} : {running_loss_rec:.5f} ] \t[ Perc_loss  : {running_loss_per:.5f} ]\
                [ GAN_loss : {opt.gan_loss_function} : {running_loss_gan:.5f} ]  \t[ overall : {running_loss_over:.5f} ] time_left: {time_left}")        
        
        # Learning rate decrease
        adjust_learning_rate(optimizer_ae, (epoch + 1), opt, opt.lr_g)

        # Save the model

        # save_model(ae_net, (epoch + 1), opt, 'autoencder_idea3_')
        save_model(ae_net, (epoch + 1), opt, 'generator_basic_ae_')
        save_model(discriminator384, (epoch + 1), opt, 'discriminator_basic_ae_')
        save_model(perceptualnet, (epoch + 1), opt, 'perceptual_basic_ae_')

        if epoch % opt.checkpoint_image== opt.checkpoint_image-1:
            utils_sample.sample_general2(opt, grayscale_without, out_with, grayscale_without, grayscale_with, opt.sample_path, (epoch + 1),saving_name=opt.saving_name+'_new')


    # logg(f'Binary mask list is : {binary_map}')
    # mean_list_masks = torch.mean(torch.stack(binary_map))
    # logg(f'mean binary mask : {mean_list_masks}')


    if opt.validation == 1:
        
        # print all model parameters
        # bottleneck network : bottleneck_net
        # autoencoder with Vessel information : ae_net2
        # encoder without Vessel information : encoder_bottle_net
        # decoder without Vessel information : decoder_net
        pytorch_total_params = sum(p.numel() for p in ae_net.parameters())
        logg(f'ae_net :{pytorch_total_params}')


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

            # # inference
            # out_enc_net1 = encoder_bottle_net(grayscale_without, mask)  # grayscale_without
            # out_bottle =  bottleneck_net(out_enc_net1) 
            # out_decoder = ae_net2.decoder(out_bottle)   
            out_ae = ae_net(grayscale_without)                      # grayscale_with

            # combination of learned information with grayscale_with
            # out_wholeimg = grayscale_with * (1 - mask) + out_decoder * mask  

            output_ae_int = (out_ae.detach().to('cpu').ravel()*255).to(torch.int16)
            # output_decoder_int = (out_decoder.detach().to('cpu').ravel()*255).to(torch.int16)
            original_with_int = (grayscale_with.detach().to('cpu').ravel()*255).to(torch.int16)

            # Validation of grayscale_without -> original image with
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

            # # Validation of grayscale_without -> original image with
            # L1_list_without2with.append(L1_Loss(out_decoder,grayscale_with ).item())
            # L2_list_without2with.append(L2_Loss(out_decoder,grayscale_with ).item())
            # Hub_list_without2with.append(Huber_Loss(out_decoder,grayscale_with ).item())
            # Focal_list_without2with.append(FocalFrequency_Loss(out_decoder,grayscale_with ).item())
            # lpips_list_without2with.append(iqa_metric_lpips(out_decoder,grayscale_with).detach().to('cpu').numpy())
            # psnr_list_without2with.append(iqa_metric_psnr(out_decoder, grayscale_with).detach().to('cpu').numpy())
            # ssim_list_without2with.append(iqa_metric_ssim(out_decoder, grayscale_with).detach().to('cpu').numpy())
            # norm_MI_without2with.append(normalized_mutual_info_score(output_decoder_int, original_with_int))
            # adj_MI_without2with.append(adjusted_mutual_info_score(output_decoder_int, original_with_int))
            # MI_without2with.append(mutual_info_score(output_decoder_int, original_with_int))
            # cor_without2with.append(np.corrcoef(output_decoder_int, original_with_int)[0][1])

            utils_sample.sample_general_bottleneck(opt, grayscale_without, out_ae, grayscale_with, grayscale_without, opt.sample_path, (1 + 1),saving_name=str(f'base_pix2pix_{batch_idx}'))


        # lpips_list_with2with = [item for sublist in lpips_list_with2with for item in sublist]
        # lpips_list_without2with = [item for sublist in lpips_list_without2with for item in sublist]
        # psnr_list_with2with = [item for sublist in psnr_list_with2with for item in sublist]
        # psnr_list_without2with = [item for sublist in psnr_list_without2with for item in sublist]
        # ssim_list_with2with = [item for sublist in ssim_list_with2with for item in sublist]
        # ssim_list_without2with = [item for sublist in ssim_list_without2with for item in sublist]


        logg(f"lpips_list_with2with is : {lpips_list_with2with}")
        # logg(f"lpips_list_without2with is : {lpips_list_without2with}")
        logg("---------------------------------------------")

        logg(f"psnr_list_with2with is : {psnr_list_with2with}")
        # logg(f"psnr_list_without2with is : {psnr_list_without2with}")
        logg("---------------------------------------------")

        logg(f"ssim_list_with2with is : {ssim_list_with2with}")
        # logg(f"ssim_list_without2with is : {ssim_list_without2with}")
        logg("---------------------------------------------")

        logg(f"L1_list_with2with are : {L1_list_with2with}")
        # logg(f"L1_list_without2with are : {L1_list_without2with}")
        logg("---------------------------------------------")

        logg(f"L2_list_with2with are : {L2_list_with2with}")
        # logg(f"L2_list_without2with are : {L2_list_without2with}")
        logg("---------------------------------------------")

        logg(f"Hub_list_with2with are : {Hub_list_with2with}")
        # logg(f"Hub_list_without2with are : {Hub_list_without2with}")
        logg("---------------------------------------------")

        logg(f"Focal_list_with2with are : {Focal_list_with2with}")
        # logg(f"Focal_list_without2with are : {Focal_list_without2with}")
        logg("---------------------------------------------")

        logg(f"norm_MI_with2with are : {norm_MI_with2with}")
        # logg(f"norm_MI_without2with are : {norm_MI_without2with}")
        logg("---------------------------------------------")

        logg(f"adj_MI_with2with are : {adj_MI_with2with}")
        # logg(f"adj_MI_without2with are : {adj_MI_without2with}")
        logg("---------------------------------------------")

        logg(f"MI_with2with are : {MI_with2with}")
        # logg(f"MI_without2with are : {MI_without2with}")
        logg("---------------------------------------------")


        logg(f"cor_with2with are : {cor_with2with}")
        # logg(f"cor_without2with are : {cor_without2with}")
        logg("---------------------------------------------")


        # AVERAGES with2with
        logg("---------------------------------------------")
        logg(f"lpips_list_with2with : {np.average(lpips_list_with2with):.6f}  \n psnr_list_with2with : {np.average(psnr_list_with2with):.6f} \n ssim_list_with2with : {np.average(ssim_list_with2with):.6f} ")
        logg(f"L1_list_with2with : {np.average(L1_list_with2with):.6f}  \n L2_list_with2with : {np.average(L2_list_with2with):.6f} \n Hub_list_with2with : {np.average(Hub_list_with2with):.6f} \n Focal_list_with2with : {np.average(Focal_list_with2with):.6f} ")

        logg(f"norm_MI_with2with : {np.average(norm_MI_with2with):.6f}  \n adj_MI_with2with : {np.average(adj_MI_with2with):.6f} \n MI_with2with : {np.average(MI_with2with):.6f} \n cor_with2with : {np.average(cor_with2with):.6f}")

        # # AVERAGES without2with
        # logg("---------------------------------------------")
        # logg(f"lpips_list_without2with : {np.average(lpips_list_without2with):.6f} \n psnr_list_without2with  : {np.average(psnr_list_without2with):.6f} \n ssim_list_without2with : {np.average(ssim_list_without2with):.6f} ")
        # logg(f"L1_list_without2with : {np.average(L1_list_without2with):.6f} \n L2_list_without2with  : {np.average(L2_list_without2with):.6f} \n Hub_list_without2with : {np.average(Hub_list_without2with):.6f} \n Focal_list_without2with : {np.average(Focal_list_without2with):.6f}")

        # logg(f"norm_MI_without2with : {np.average(norm_MI_without2with):.6f}  \n adj_MI_without2with : {np.average(adj_MI_without2with):.6f} \n MI_without2with : {np.average(MI_without2with):.6f} \n cor_without2with : {np.average(cor_without2with):.6f}")

        quit()