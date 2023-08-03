################ imports ################
import numpy as np
import os
import cv2
import torchvision
import torch
from PIL import Image

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def sample(opt, grayscale, mask, out, unmasked_out, save_folder, epoch, saving_name):
    # to cpu
    grayscale = grayscale[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    mask = mask[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    out = out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)  
    unmasked_out = unmasked_out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    ### CHANGE THIS!
    # process
    masked_img = grayscale * (1 - mask) + mask                                                  # 256 * 256 * 1
    masked_img = np.concatenate((masked_img, masked_img, masked_img), axis = 2)                 # 256 * 256 * 3 (√)
    masked_img = (masked_img * 255).astype(np.uint8)
    grayscale = np.concatenate((grayscale, grayscale, grayscale), axis = 2)                     # 256 * 256 * 3 (√)
    grayscale = (grayscale * 255).astype(np.uint8)
    mask = np.concatenate((mask, mask, mask), axis = 2)                                         # 256 * 256 * 3 (√)
    mask = (mask * 255).astype(np.uint8)
    
    out = np.concatenate((out, out, out), axis = 2)                                             # 256 * 256 * 3 (√)
    out = (out * 255).astype(np.uint8)

    unmasked_out = np.concatenate((unmasked_out, unmasked_out, unmasked_out), axis = 2)                                             # 256 * 256 * 3 (√)
    unmasked_out = (unmasked_out * 255).astype(np.uint8)
    print("unmasked_out shape : ", unmasked_out.shape)
    print("grayscale shape : ", grayscale.shape)
    print("mask shape : ", mask.shape)
    print("masked_img shape : ", masked_img.shape)
    # save
    img = np.concatenate((grayscale, mask, masked_img, out, unmasked_out), axis = 1)
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    cv2.imwrite(imgname, img)


def sample2(opt, image, image_pair,  output, save_folder, epoch, saving_name):
    # to cpu
    image = image[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    image_pair = image_pair[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    output = output[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)  
    # unmasked_out = unmasked_out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    # masked_img = grayscale * (1 - mask) + mask                                                  # 256 * 256 * 1
    # masked_img = np.concatenate((masked_img, masked_img, masked_img), axis = 2)                 # 256 * 256 * 3 (√)
    # masked_img = (masked_img * 255).astype(np.uint8)
    image = np.concatenate((image, image, image), axis = 2)                     # 256 * 256 * 3 (√)
    image = (image * 255).astype(np.uint8)
    image_pair = np.concatenate((image_pair, image_pair, image_pair), axis = 2)                                         # 256 * 256 * 3 (√)
    image_pair = (image_pair * 255).astype(np.uint8)
    
    output = np.concatenate((output, output, output), axis = 2)                                             # 256 * 256 * 3 (√)
    output = (output * 255).astype(np.uint8)

    # unmasked_out = np.concatenate((unmasked_out, unmasked_out, unmasked_out), axis = 2)                                             # 256 * 256 * 3 (√)
    # unmasked_out = (unmasked_out * 255).astype(np.uint8)
    # save
    img = np.concatenate((image, image_pair, output), axis = 1)
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    cv2.imwrite(imgname, img)

def sample3(opt, image, image_pair, output, output2, save_folder, epoch, saving_name):
    # to cpu
    image = image[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    image_pair = image_pair[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    output = output[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)  
    output2 = output2[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)  
    # unmasked_out = unmasked_out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    # masked_img = grayscale * (1 - mask) + mask                                                  # 256 * 256 * 1
    # masked_img = np.concatenate((masked_img, masked_img, masked_img), axis = 2)                 # 256 * 256 * 3 (√)
    # masked_img = (masked_img * 255).astype(np.uint8)
    image = np.concatenate((image, image, image), axis = 2)                     # 256 * 256 * 3 (√)
    image = (image * 255).astype(np.uint8)
    image_pair = np.concatenate((image_pair, image_pair, image_pair), axis = 2)                                         # 256 * 256 * 3 (√)
    image_pair = (image_pair * 255).astype(np.uint8)
    
    output = np.concatenate((output, output, output), axis = 2)                                             # 256 * 256 * 3 (√)
    output = (output * 255).astype(np.uint8)

    output2 = np.concatenate((output2, output2, output2), axis = 2)                                             # 256 * 256 * 3 (√)
    output2 = (output2 * 255).astype(np.uint8)
    # unmasked_out = np.concatenate((unmasked_out, unmasked_out, unmasked_out), axis = 2)                                             # 256 * 256 * 3 (√)
    # unmasked_out = (unmasked_out * 255).astype(np.uint8)
    # save
    img = np.concatenate((image, image_pair, output, output2), axis = 1)
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    cv2.imwrite(imgname, img)



def sample_general(opt, img, mask, combined_out, unmasked_out, save_folder, epoch, saving_name):
    # to cpu

    # print("img shape : ", img.shape)
    # print("mask shape : ", mask.shape)
    # print("combined_out shape : ", combined_out.shape)
    # print("unmasked_out shape : ", unmasked_out.shape)
    in_dims , out_dims = 1,1
    if opt.in_channels == 1:
        in_dims = 3
    if opt.out_channels == 1:
        out_dims = 3

    img = torch.cat((img[0],)*in_dims, 0)
    mask = torch.cat((mask[0],)*3, 0)
    combined_out = torch.cat((combined_out[0],)*out_dims, 0)
    unmasked_out = torch.cat((unmasked_out[0],)*out_dims, 0)
    
    masked_img = img * (1 - mask) + mask

    test_list = [img, mask, masked_img , combined_out, unmasked_out]
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    torchvision.utils.save_image(test_list, imgname)

def sample_general2(opt, img, mask, combined_out, unmasked_out, save_folder, epoch, saving_name):
    # to cpu

    # print("img shape : ", img.shape)
    # print("mask shape : ", mask.shape)
    # print("combined_out shape : ", combined_out.shape)
    # print("unmasked_out shape : ", unmasked_out.shape)
    in_dims , out_dims = 1,1
    if opt.in_channels == 1:
        in_dims = 3
    if opt.out_channels == 1:
        out_dims = 3

    img = torch.cat((img[0],)*in_dims, 0)
    mask = torch.cat((mask[0],)*3, 0)
    combined_out = torch.cat((combined_out[0],)*out_dims, 0)
    unmasked_out = torch.cat((unmasked_out[0],)*out_dims, 0)
    
    # masked_img = img * (1 - mask) + mask

    test_list = [img, mask,  combined_out, unmasked_out]
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    torchvision.utils.save_image(test_list, imgname)


def sample_Y(opt, img, out, save_folder, epoch, saving_name):
    # to cpu

    # print("img shape : ", img.shape)
    # print("mask shape : ", mask.shape)
    # print("combined_out shape : ", combined_out.shape)
    # print("unmasked_out shape : ", unmasked_out.shape)
    in_dims , out_dims = 1,1
    if opt.in_channels == 1:
        in_dims = 3
    if opt.out_channels == 1:
        out_dims = 3

    img = torch.cat((img[0],)*in_dims, 0)
    out = torch.cat((out[0],)*3, 0)
    # combined_out = torch.cat((combined_out[0],)*out_dims, 0)
    # unmasked_out = torch.cat((unmasked_out[0],)*out_dims, 0)
    
    # masked_img = img * (1 - mask) + mask

    test_list = [img, out]
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    torchvision.utils.save_image(test_list, imgname)



def sample_general_bottleneck(opt, img, mask, combined_out, unmasked_out, save_folder, epoch, saving_name):
    # to cpu

    # print("img shape : ", img.shape)
    # print("mask shape : ", mask.shape)
    # print("combined_out shape : ", combined_out.shape)
    # print("unmasked_out shape : ", unmasked_out.shape)
    in_dims , out_dims = 1,1
    if opt.in_channels == 1:
        in_dims = 3
    if opt.out_channels == 1:
        out_dims = 3

    img = torch.cat((img[0],)*in_dims, 0)
    mask = torch.cat((mask[0],)*in_dims, 0)
    combined_out = torch.cat((combined_out[0],)*out_dims, 0)
    unmasked_out = torch.cat((unmasked_out[0],)*out_dims, 0)
    

    test_list = [img, mask, combined_out, unmasked_out]
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    torchvision.utils.save_image(test_list, imgname)


def sample_one(opt, img, save_folder,epoch, saving_name):
    
    in_dims , out_dims = 1,1
    if opt.in_channels == 1:
        in_dims = 3
    if opt.out_channels == 1:
        out_dims = 3
    # print(type(img))
    img = torch.cat((img[0],)*in_dims, 0)
    img = torchvision.transforms.Resize((int(opt.imgsize*1.5),opt.imgsize))(img)
    # print(type(img))
    test_list = [img]
    imgname = os.path.join(save_folder,str(opt.model) +  '_' + str(saving_name) + '_' + str(opt.loss_function) + '_' + str(opt.gan_loss_function) + '_' + str(epoch)+ '_'  + str(opt.mask_type) + '.png')
    torchvision.utils.save_image(test_list, imgname)
