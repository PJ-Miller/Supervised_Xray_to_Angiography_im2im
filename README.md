# Supervised_Xray_to_Angiography_im2im
Repo of different models and frameworks that make use image-to-image translation in order to recreate angiographic images from X-rays.


# overall

this was implemented to run on the polyaxon cluster.
For local run, change will be made soon

# requirements
in requirements.txt


# description
depending on which model to train

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



- hyperparameters: \\ to do

        options:
            -- 