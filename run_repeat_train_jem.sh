!/bin/bash
pip install -r requirements.txt ;
for i in `seq 1`
do
    # Train
    python train_jem.py \
    --basenet wideresnet \
    --lr .0001 \
    --dataset bloodmnist \
    --n_classes 8 \
    --optimizer adam \
    --p_x_weight 1.0 \
    --p_y_given_x_weight 1.0 \
    --p_x_y_weight 0.0 \
    --sigma .03 \
    --save_dir jem_bloodmnist/seed$i/ \
    --warmup_iters 1000 \
    --buffer_size 10000 \
    --n_epochs 100 \
    --n_ch 3 \
    --batch_size 128 \
    --im_sz 28 \
    --labels_per_class 10 \
    --semisupervision_seed $i \
    --print_to_log ;
    # Test
    for j in `seq 9`
    do
        python test_jem.py \
        --basenet wideresnet \
        --dataset bloodmnist \
        --n_classes 8 \
        --load_path jem_bloodmnist/seed$i/best_valid_ckpt_alit$j.pt \
        --n_ch 3 \
        --batch_size 128 \
        --im_sz 28 \
        --save_dir jem_bloodmnist/seed$i/clf_alit$j/ \
        --print_to_log ;
    done
    # python test_jem.py \
    # --basenet wideresnet \
    # --dataset bloodmnist \
    # --n_classes 8 \
    # --load_path jem_bloodmnist/seed$i/best_valid_ckpt_alit$i.pt \
    # --n_ch 3 \
    # --batch_size 128 \
    # --im_sz 28 \
    # --save_dir jem_bloodmnist/seed$i/clf/ \
    # --print_to_log ;
done
