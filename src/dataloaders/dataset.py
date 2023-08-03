################ imports ################
import os
import math
import random
import torch
import torchvision
import torchvision.transforms.functional as TF
# from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
from ..utils import utils_core
from ..utils.utils_core import logg
from skimage.exposure import match_histograms
import skimage
# BOOLCONTRAST = ['True', 'False']
ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']


# Inpainting Dataset class that delivers paired data img, img_pairs from folders opt.baseroot and opt.baseroot_pair in arguments
class InpaintDataset_pairs(Dataset):
    def __init__(self, opt, transforms):
        assert opt.mask_type in ALLMASKTYPES
        # assert opt.contrast in BOOLCONTRAST
        self.opt = opt
        self.imglist_1 = utils_core.get_jpgs(opt.baseroot) # .sort()
        self.imglist_2 = utils_core.get_jpgs(opt.baseroot_pair) # .sort()
        # print(self.imglist_1,self.imglist_2)
        self.transform = transforms
        

    def __len__(self):
        return len(self.imglist_1) 

    def __getitem__(self, index):

        imgname_1 = self.imglist_1[index]    
        imgname_2 = self.imglist_2[index]                                   # name of one image


        imgpath_1 = os.path.join(self.opt.baseroot, imgname_1)  
        imgpath_2 = os.path.join(self.opt.baseroot_pair, imgname_2)  

        img = torchvision.io.read_image(imgpath_1)
        img_pair = torchvision.io.read_image(imgpath_2)

        seed = np.random.randint(2147483647)
        random.seed(seed) 
        torch.manual_seed(seed)

        if self.transform:
            img = self.transform(img)
            # test without
            
            

        random.seed(seed) 
        torch.manual_seed(seed)      
        if self.transform:    
            img_pair = self.transform(img_pair)
            # img = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=7.)(img)

        # if self.transform:
        #     img = self.transform(img)            
        #     img_pair = self.transform(img_pair)
        #     # img_pair = img
        #     # print(img[0][0][0:10])
        #     # print(img_pair[0][0][0:10])

        #     # img = torchvision.transforms.CenterCrop((self.opt.imgsize,self.opt.imgsize))(img)
        #     # img_pair = torchvision.transforms.CenterCrop((self.opt.imgsize,self.opt.imgsize))(img_pair)

        #     # torchvision.utils.save_image(img[0], 'img1.png')
        #     # torchvision.utils.save_image(img_pair[0], 'img2.png')
        #     # both_images = torch.cat((img.unsqueeze(0), img_pair.unsqueeze(0)),0)
        #     # both_images = torchvision.transforms.CenterCrop((self.opt.imgsize,self.opt.imgsize))(both_images)
        #     # print(both_images.shape)
        #     # img = both_images[0]
        #     # img_pair = both_images[1]
        #     # logg(f"path 1 : , {imgpath_1}",verbose=False)
        #     # logg(f"path 2 : , {imgpath_2}",verbose=False)
        #     # print(img.shape)

            
        #     # resize = torchvision.transforms.Resize(size=(300, 300))
        #     # img = resize(img)
        #     # img_pair = resize(img_pair)

        #             # Random crop
        #     # i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        #     #     img, output_size=(self.opt.imgsize, self.opt.imgsize))
        #     # img = TF.crop(img, i, j, h, w)
        #     # img_pair = TF.crop(img_pair, i, j, h, w)
        #     # logg(f"parameters i, j, h, w are : , {i}, {j}, {h}, {w}",verbose=False)
        #     # logg(f"newest img shape is , {img.shape}, {img_pair.shape}",verbose=False)
        #     # logg(f"path 1 : , {imgpath_1}",verbose=False)
        #     # logg(f"path 2 : , {imgpath_2}",verbose=False)
        #     # img = TF.to_tensor(img)
        #     # img_pair = TF.to_tensor(img_pair)


        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)

        mask = torch.from_numpy(mask).contiguous()

        img = img/255.0
        img_pair = img_pair/255.0
    
        # torchvision.utils.save_image(img[0], 'img1.png')
        # torchvision.utils.save_image(img_pair[0], 'img2.png')
        ## transformation/normalizations according to dataset mean, sd // optional
        # img =torchvision.transforms.Normalize((self.opt.baseroot_mean,), (self.opt.baseroot_std,))(img)      
        # img_pair =torchvision.transforms.Normalize((self.opt.baseroot_pair_mean,), (self.opt.baseroot_pair_std,))(img_pair)
            
        ## contrast option in arguments
        # no contrast // default
        if self.opt.contrast == 1:
            return img, img_pair, mask
        # random contrast always on 
        elif self.opt.contrast == 2:
            change = torch.randint(0,0.1,(1,))
            img = torch.add(img,change)
            img_pair = torch.add(img_pair,change)
            return img, img_pair, mask
        # probability of contrast 0.5, change of changing contrast
        elif  self.opt.contrast == 3:
            if np.random.binomial(1,p= 0.5) ==1 :
                return img, img_pair, mask
            else:
                contrast = round(random.uniform(0.1,5),2)
                img = torchvision.transforms.functional.adjust_contrast(img, contrast)
                img_pair = torchvision.transforms.functional.adjust_contrast(img_pair, contrast)
                return img, img_pair, mask
        else:
            print("ERROR!!!")
            quit()
        


    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)





class InpaintDataset_pairs_without_mask(Dataset):
    def __init__(self, opt, transforms):
        assert opt.mask_type in ALLMASKTYPES
        # assert opt.contrast in BOOLCONTRAST
        self.opt = opt
        self.imglist_1 = utils_core.get_jpgs(opt.baseroot) # .sort()
        self.imglist_2 = utils_core.get_jpgs(opt.baseroot_pair) # .sort()
        # print(self.imglist_1,self.imglist_2)
        self.transform = transforms
        

    def __len__(self):
        return len(self.imglist_1) 

    def __getitem__(self, index):

        imgname_1 = self.imglist_1[index]    
        imgname_2 = self.imglist_2[index]                                   # name of one image


        imgpath_1 = os.path.join(self.opt.baseroot, imgname_1)  
        imgpath_2 = os.path.join(self.opt.baseroot_pair, imgname_2)  

        img = torchvision.io.read_image(imgpath_1)
        img_pair = torchvision.io.read_image(imgpath_2)

        seed = np.random.randint(2147483647)
        random.seed(seed) 
        torch.manual_seed(seed)

        if self.transform:
            img = self.transform(img)
            # test without
            
            

        random.seed(seed) 
        torch.manual_seed(seed)      
        if self.transform:    
            img_pair = self.transform(img_pair)
            # img = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=7.)(img)


        img = img/255.0
        img_pair = img_pair/255.0
    
        return img, img_pair
    


class InpaintDataset_pairs_without_mask_vadilation(Dataset):
    def __init__(self, opt, transforms):
        assert opt.mask_type in ALLMASKTYPES
        # assert opt.contrast in BOOLCONTRAST
        self.opt = opt
        self.imglist_1 = utils_core.get_jpgs(opt.baseroot) # .sort()
        self.imglist_2 = utils_core.get_jpgs(opt.baseroot_pair) # .sort()
        print(self.imglist_1)
        self.transform = transforms
        

    def __len__(self):
        return len(self.imglist_1) 

    def __getitem__(self, index):

        imgname_1 = self.imglist_1[index]    
        imgname_2 = self.imglist_2[index]                                   # name of one image


        imgpath_1 = os.path.join(self.opt.baseroot, imgname_1)  
        imgpath_2 = os.path.join(self.opt.baseroot_pair, imgname_2)  

        img = torchvision.io.read_image(imgpath_1)
        img_pair = torchvision.io.read_image(imgpath_2)

        seed = np.random.randint(2147483647)
        random.seed(seed) 
        torch.manual_seed(seed)

        
        if self.transform:
            if img.shape[0] > 1 :
                print(img.shape)
                img = img[0:1]
            img = self.transform(img)


            # img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1) 
            # test without
            
        random.seed(seed) 
        torch.manual_seed(seed)      
        if self.transform:    
            if img_pair.shape[0] > 1 :
                img_pair = img[0:1]
            img_pair = self.transform(img_pair)

        img[(img <= 70)]=70
        
        img = skimage.exposure.equalize_adapthist(img.numpy()/img.max().numpy())
        img = torch.from_numpy(img).float() 

        img_pair = (img_pair/255.0)
        # print(img.shape, img_pair.shape)

        return img, img_pair
    
    

# Inpainting Dataset class that delivers paired data img, img_pairs from folders opt.baseroot and opt.baseroot_pair in arguments
class InpaintDataset_pairs_validation(Dataset):
    def __init__(self, opt, transforms):
        assert opt.mask_type in ALLMASKTYPES
        # assert opt.contrast in BOOLCONTRAST
        self.opt = opt
        self.imglist_1 = utils_core.get_jpgs(opt.baseroot) # .sort()
        self.imglist_2 = utils_core.get_jpgs(opt.baseroot_pair) # .sort()
        # print(self.imglist_1,self.imglist_2)
        self.transform = transforms
        

    def __len__(self):
        return len(self.imglist_1) 

    def __getitem__(self, index):

        imgname_1 = self.imglist_1[index]    
        imgname_2 = self.imglist_2[index]                                   # name of one image


        imgpath_1 = os.path.join(self.opt.baseroot, imgname_1)  
        imgpath_2 = os.path.join(self.opt.baseroot_pair, imgname_2)  

        img = torchvision.io.read_image(imgpath_1)
        img_pair = torchvision.io.read_image(imgpath_2)

        seed = np.random.randint(2147483647)
        random.seed(seed) 
        torch.manual_seed(seed)

        if self.transform:
            if img.shape[0] > 1 :
                print(img.shape)
                img = img[0:1]
            img = self.transform(img)

            

        random.seed(seed) 
        torch.manual_seed(seed)      
        if self.transform:    
            if img_pair.shape[0] > 1 :
                img_pair = img[0:1]
            img_pair = self.transform(img_pair)


        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)

        mask = torch.from_numpy(mask).contiguous()

        img[(img <= 70)]=70
        
        img = skimage.exposure.equalize_adapthist(img.numpy()/img.max().numpy())
        img = torch.from_numpy(img).float() 

        img_pair = (img_pair/255.0)
    
        torchvision.utils.save_image(img[0], 'img1.png')
        torchvision.utils.save_image(img_pair[0], 'img2.png')

        if self.opt.contrast == 1:
            return img, img_pair, mask
        # random contrast always on 
        elif self.opt.contrast == 2:
            change = torch.randint(0,0.1,(1,))
            img = torch.add(img,change)
            img_pair = torch.add(img_pair,change)
            return img, img_pair, mask
        # probability of contrast 0.5, change of changing contrast
        elif  self.opt.contrast == 3:
            if np.random.binomial(1,p= 0.5) ==1 :
                return img, img_pair, mask
            else:
                contrast = round(random.uniform(0.1,5),2)
                img = torchvision.transforms.functional.adjust_contrast(img, contrast)
                img_pair = torchvision.transforms.functional.adjust_contrast(img_pair, contrast)
                return img, img_pair, mask
        else:
            print("ERROR!!!")
            quit()
        


    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    