#!/bin/bash
#model_save_path=covidx_gan
#if [ ! -d $model_save_path ]
#then
#    mkdir $model_save_path
#fi


python run_train.py  
                    # --gpu=<id> \
                    # --view=<dset> \
                    
                    # --resume_c_w="/shared/anastasio5/COVID19/covid19_classification_with_gan/covidx_train/gan_res50_224_patch_ssim/best_model.pt" \
                    # --resume_d_w="/shared/anastasio5/COVID19/covid19_classification_with_gan/covidx_train/gan_res50_224_patch_ssim/D_epoch_200.pt" \
                    # resent 50 pretrained weights