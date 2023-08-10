# Supervised_Xray_to_Angiography_im2im
Repo of different models and frameworks that make use image-to-image translation in order to recreate angiographic images from X-rays.


# overall

this was implemented to run on the polyaxon cluster.

needs: polyaxon installed


change the hyperparameters in polyaxonfile.yaml 
choose function to run in polyaxonfile.yaml 

run in console:
    polyaxon run -f polyaxonfile.yaml -u -l


changes of hyperparameters happen in the yaml file: 


# requirements
in requirements.txt


# description
depending on which model to train in yaml file

- choose model to train models : 
        options:

            train_prototype_4 : pretraining of autoencoder -> train of bottleneck --> loss function evaluation between pretraining ae output and model output
            train_prototype_4Y : pretraining of autoencoder -> train of bottleneck --> loss function evaluation between ground truth and model output
            train_prototype 4Y_opt : a bit optimized performance 
            train_prototype_4Y_pix2pix: adds perceptual and discriminator network to prototype 4Y
            train_prototype_3 : Deprecated ( will see what to do with this )
            train_prototype_2 : Deprecated ( will see what to do with this )
            train_prototype_1 : Deprecated ( will see what to do with this )
            train_prototype_5 : same as prototype 4 but with different bottleneck options // must be changed
             
            train_basic_autoencoder : run basic autoencoder
            train_basic_pix2pix : run basic autoencoder and adds perceptual and discriminator network



- hyperparameters: 

        used options in the  polyaxonyaml.file :
            --saving_name               # saving name of images
            --checkpoint_interval       # save models every x epochs
            --finetune_path_encoder     # pre-trained encoder network place. If not empty, will use this network 
            --finetune_path_decoder     # pre-trained decoder network place. If not empty, will use this network 
            --finetune_path_autoencoder # pre-trained autoencoder network place. If not empty, will use this network 
            --finetune_path_bottleneck  # pre-trained additional bottleneck network place. If not empty, will use this network 
            --save_path             # path of saving 
            --sample_path           # path to save samples
            --log_path              # path for logging
            --epoch                 # number of epochs
            --batch_siz             # number of batchsize
            --lr_g                  # learning rate generator
            --lr_decrease_epoch     # learning rate decay rate
            --lr_decrease_factor    # learning rate decrease factor
            --num_workers           # number of workers
            --checkpoint_image      # sample images every x epochs
            --latent_channels       # defines the number of latent channels
            --latent_multi          # another way to change the latent channels
            --baseroot              # root pair training data without vessel information
            --baseroot_pair         # root pair training data with vessel information
            --imgsize               # size of images (size,size)
            --mask_type             # format of images : can be "free_from" or "bbox, single_bbox, full_box"
            --margin                # margin from mask to image limits
            --mask_num              # number of masks
            --bbox_shape            # size of box masks
            --max_angle             # maximum angle of free_form masks
            --max_len               # maximum length of free_form masks
            --max_width             # maximum width of free_form masks
            --contrast              # contrast options
            --model                 # can change architectures in older models
            --loss_function         # Loss function of reconstruction loss
            --gan_loss_function     # Generator Loss function
            --log_interval          # interval between logs
            --weight_decay          # weight_decay hyperparameter
            --test_base             # root pair test data without vessel information
            --test_base_pair        # root pair test data with vessel information 


        other options are in the each training file in the arguments.
