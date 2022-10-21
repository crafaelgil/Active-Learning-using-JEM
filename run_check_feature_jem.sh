#!/bin/bash
python check_feature_jem.py \
--basenet wideresnet \
--dataset MRI \
--n_classes 2 \
--n_ch 3 \
--batch_size 64 \
--im_sz 32 \
--load_path jem_MRI/best_valid_ckpt.pt \
--save_dir jem_MRI/feature/ \
--print_to_log
