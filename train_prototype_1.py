################ imports ################
import argparse
import logging
import datetime
# from src.models import prototype_1 
from src.utils.utils_core import logg
from  src.models.prototype_1 import Prototype_1
# from  src.validations.validation_prototype_1 import Validation_Prototype_1


import os
import pyiqa
# Polyaxon
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

if __name__ == "__main__":



    # ---------------------------------------------------------------------------------------#
    #                               Initialize the parameters
    # ---------------------------------------------------------------------------------------#
    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------------------------------------#
    # ------------------------        General parameters            ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--pre_train', type = int, default = 0, \
        help = '0 pretrains models generator, 1 trains model, 2 pretrains and trains') # needs to be done
    parser.add_argument('--saving_name', type = str, default = "prototype_1", \
        help = 'extra string to save model and pictures')
    parser.add_argument('--multi_gpu', type = bool, default = False, \
        help = 'nn.Parallel on or not')
    # parser.add_argument('--gpu_ids', type = str, default = "0, 1", \
    #     help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, \
        help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 200, \
        help = 'interval between model checkpoints')
    parser.add_argument('--validation', type= int, default=1, \
        help = 'if 1 then model will test validation test set')

    # ---------------------------------------------------------------------------------------#
    # ------------------------           Path parameters            ------------------------ #
    # ---------------------------------------------------------------------------------------#
       
    parser.add_argument('--finetune_path_encoder', type = str, default= "" , \
    # default = 'D:\\Thesis_Repo\\GhostModel\\new_net\\models\\Model_version_SECOND_TESTTTT_RUN_TWO_encBot_new_2_L1_L2_epoch220_batchsize1.pth', \
    #".\models\Model_encBot_L1_L2_epoch420_batchsize1.pth", \
        help = 'load name path of models')
    parser.add_argument('--finetune_path_decoder', type = str, default= "",  \
    # default = 'D:\\Thesis_Repo\\GhostModel\\new_net\\models\\Model_version_SECOND_TESTTTT_RUN_TWO_dec_new_2_L1_L2_epoch220_batchsize1.pth', \
    # ".\models\Model_dec_L1_L2_epoch420_batchsize1.pth", \
        help = 'load name path of models')
    parser.add_argument('--finetune_path_autoencoder', type = str, default= "", \
    # default = 'D:\\Thesis_Repo\\GhostModel\\new_net\\models\\Model_version_SECOND_TESTTTT_RUN_TWO_autoen2_new_2_L1_L2_epoch220_batchsize1.pth', \
    # ".\models\Model_autoen2_L1_L2_epoch420_batchsize1.pth", \
        help = 'load name path of models')
    parser.add_argument('--finetune_path_bottleneck', type = str, default= "", \
    # default = 'D:\\Thesis_Repo\\GhostModel\\new_net\\models\\Model_version_original_bottleneck_version_2__Bottleneck_new_3__L1_L2_epoch420_batchsize1.pth', \
    # ".\models\Model_autoen2_L1_L2_epoch420_batchsize1.pth", \
        help = 'load name path of models')
    parser.add_argument('--finetune_path', type = str, default = "", \
        help = 'load name path of models')
    parser.add_argument('--save_path', type = str, default = './models', \
        help = 'path of folder to save model')
    parser.add_argument('--sample_path', type = str, default = './samples', \
        help = 'path to folder to save training samples')
    parser.add_argument('--log_path', type = str, default = './log', \
        help = 'path to folder to save log files')            
    
    # ---------------------------------------------------------------------------------------#
    # ------------------------        Training parameters           ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--epochs', type = int, default = 400, \
        help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, \
        help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 2e-4, \
        help = 'Adam: learning rate generator')
    parser.add_argument('--lr_d', type = float, default = 2e-4, \
        help = 'Adam: learning rate discriminator')
    parser.add_argument('--b1', type = float, default = 0.5, \
        help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, \
        help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, \
        help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 25, \
        help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, \
        help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_loss_function', type = float, default = 20, \
        help = 'the parameter of reconstruction Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 1, \
        help = 'the parameter of perceptual loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.1, \
        help = 'the parameter of GAN loss')
    parser.add_argument('--num_workers', type = int, default = 0, \
        help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint_image', type = int, default = 50, \
        help = 'interval between saved validation images')
    # ---------------------------------------------------------------------------------------#
    # ------------------------        Network parameters            ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--in_channels', type = int, default = 1, \
        help = 'input image channel')
    parser.add_argument('--out_channels', type = int, default = 1, \
        help = 'output image channel')
    parser.add_argument('--mask_channels', type = int, default = 1, \
        help = 'input mask channels')
    parser.add_argument('--latent_channels', type = int, default = 64, \
        help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'reflect', \
        help = 'the padding type, choices reflect, replicate, zero')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', \
        help = 'the activation type, choices relu, lrelu, prelu, selu, tanh, sigmoid, none')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', \
        help = 'the activation type, choices relu, lrelu, prelu, selu, tanh, sigmoid, none')
    parser.add_argument('--norm_g', type = str, default = 'in', \
        help = 'normalization type, choices bn, in, ln, none')
    parser.add_argument('--norm_d', type = str, default = 'bn', \
        help = 'normalization type, choices bn, in, ln, none')
    parser.add_argument('--init_type', type = str, default = 'normal', \
        help = 'the initialization type, choices normal, xavier, kaiming, orthogonal')
    parser.add_argument('--init_gain', type = float, default = 0.02, \
        help = 'the initialization gain')
    parser.add_argument('--latent_multi',  type=int, default=4, \
        help='multiplier of the latent space in bottleneck architecture')
    
    # ---------------------------------------------------------------------------------------#
    # ------------------------        Dataset parameters            ------------------------ #
    # ---------------------------------------------------------------------------------------#
# D:\Data\pseudoDDR_versions\Dongyang_DRR\D1\withoutVessel
# D:\Data\pseudoDDR_versions\Dongyang_DRR\D1\withVessel
    parser.add_argument('--baseroot', type = str, default = "D:\\Data\\pseudoDDR_versions\\Dongyang_DRR\\D1\\withoutVessel_1\\", \
        help = 'the base training folder for inpainting network')
    parser.add_argument('--baseroot_pair', type = str, default = "D:\\Data\\pseudoDDR_versions\\Dongyang_DRR\\D1\\withVessel_1\\", \
        help = 'the base training folder for inpainting network')
    parser.add_argument('--test_base', type = str, default = "D:\\Data\\pseudoDDR_versions\\Dongyang_DRR\\D1\\withoutVessel_1\\", \
        help = 'the base training folder for inpainting network')
    parser.add_argument('--test_base_pair', type = str, default = "D:\\Data\\pseudoDDR_versions\\Dongyang_DRR\\D1\\withVessel_1\\", \
        help = 'the base training folder for inpainting network')    
    # parser.add_argument('--baseroot', type = str, default = "D:\\Thesis_Repo\\data\\with_shapes\\train_2\\224\\", \
    #     help = 'the base training folder for inpainting network')
    # parser.add_argument('--baseroot_pair', type = str, default = "D:\\Thesis_Repo\\data\\with_shapes\\train\\comb_not_blur\\224\\", \
    #     help = 'the base training folder for inpainting network')
    parser.add_argument('--imgsize', type = int, default = 224, \
        help = 'size of image')
    parser.add_argument('--baseroot_mean', type = float, default = 0.2173, \
        help = 'the base training datasets mean')
    parser.add_argument('--baseroot_std', type = float, default = 0.3274, \
        help = 'the base training datasets std') 
    parser.add_argument('--baseroot_pair_mean', type = float, default = 0.1749, \
        help = 'the paired base training datasets mean')
    parser.add_argument('--baseroot_pair_std', type = float, default = 0.2108 , \
        help = 'the paired base training datasets std')   

    # ---------------------------------------------------------------------------------------#
    # ------------------------          mask parameters             ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--mask_type', type = str, default = 'free_form', \
        help = 'mask type, choices bbox, single_bbox, free_form, full_box')
    parser.add_argument('--margin', type = int, default = 10, \
        help = 'margin of box masks to image egdes')
    parser.add_argument('--mask_num', type = int, default = 20, \
        help = 'number of masks')
    parser.add_argument('--bbox_shape', type = int, default = 30, \
        help = 'height and width of box masks')
    parser.add_argument('--max_angle', type = int, default = 4, \
        help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, \
        help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 2, \
        help = 'parameter of width for free form mask')
    parser.add_argument('--contrast', type = int, default = 1, \
        help = 'parameter of random contrast change on 0 /off 1 ')


    # ---------------------------------------------------------------------------------------#
    # ------------------------          model parameters            ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--model', type = int, default = 0, \
        help = 'uses different AE CNN architecture original = 0')
    parser.add_argument('--loss_function', type = str, default = 'L1', \
        help = 'loss function used in training of generator, choices L1, L2, Huber, Focal')
    parser.add_argument('--gan_loss_function', type = str, default = 'L2', \
        help = 'loss function used in training of generator, choices L1, L2, Huber, Focal')
    parser.add_argument('--groups', type = int, default = 1, \
        help = ' ')

    # ---------------------------------------------------------------------------------------#
    # ------------------------           log parameters             ------------------------ #
    # ---------------------------------------------------------------------------------------#

    parser.add_argument('--log_interval', type=int, default=5, \
        help='how many batches to wait before logging training status')


    # args, unknown = parser.parse_known_args()
    opt = parser.parse_args()
    # print(opt)
    # Polyaxon
    experiment = Experiment()

    data_dir = os.path.join(list(get_data_paths().values())[0])
    data_paths = get_data_paths()
    outputs_path = get_outputs_path()


    # new paths
    opt.baseroot = data_paths['data1'] + opt.baseroot 
    opt.baseroot_pair = data_paths['data1'] + opt.baseroot_pair
    opt.test_base = data_paths['data1'] + opt.test_base 
    opt.test_base_pair = data_paths['data1'] + opt.test_base_pair

    # opt.finetune_path_encoder = data_paths['data1'] + opt.finetune_path_encoder
    # opt.finetune_path_decoder = data_paths['data1'] + opt.finetune_path_decoder
    # opt.finetune_path_autoencoder = data_paths['data1'] + opt.finetune_path_autoencoder
    # opt.finetune_path_bottleneck = data_paths['data1'] + opt.finetune_path_bottleneck


    opt.save_path = outputs_path + opt.save_path
    opt.sample_path = outputs_path + opt.sample_path
    opt.log_path = outputs_path + opt.log_path


    # volume_path = data_paths['data1'] + opt.input

    print(data_dir, data_paths, outputs_path)
    print(opt.baseroot, opt.baseroot_pair)
    print(opt.save_path, opt.sample_path,opt.log_path)


    # logging.basicConfig(
    #     filename=f'{opt.log_path}/{opt.pre_train}_{opt.saving_name}_{datetime.datetime.now().date().isoformat()}_.log', 
    #     encoding='utf-8',
    #     level=logging.DEBUG)


    logg('#-'*40,verbose=True)
    logg('Namespace variables', verbose=True)
    for item in  vars(opt).items():
        logg(f'\t--{item[0]}\t {item[1]}', verbose=True)

    Prototype_1(opt)
