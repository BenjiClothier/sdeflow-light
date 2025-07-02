CUDA_VISIBLE_DEVICES=1 python train_img.py --expname=mnist_bpd \
    --dataset=mnist --print_every=2000 --sample_every=2000 --checkpoint_every=2000 --num_steps=1000 \
    --batch_size=1280 --lr=0.0001 --num_iterations=100000 --real=True --debias=False --track_bpd_steps=True \
    --save_bpd_trajectory=True 