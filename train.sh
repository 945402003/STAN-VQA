 CUDA_VISIBLE_DEVICES=0 python -u train.py \
 --database LSVQ \
 --model_name UGC_BVQA_model \
 --conv_base_lr 0.00001 \
 --epochs 10 \
 --train_batch_size 8 \
 --print_samples 1000 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --exp_version 0 \
 --loss_type L1RankLoss \
 --resize 520 \
 --crop_size 448 \
 >> logs/train_UGC_BVQA_model_L1RankLoss_resize_520_crop_size_448_exp_version_0.log


CUDA_VISIBLE_DEVICES="1" python -u train_KoNViD-1k.py \
 --database KoNViD-1k \
 --model_name UGC_BVQA_model \
 --conv_base_lr 0.00001 \
 --epochs 200 \
 --train_batch_size 4 \
 --print_samples 1000 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --exp_version 0 \
 --loss_type L1RankLoss \
 --resize 520 \
 --crop_size 448 \
 >> logs/train_UGC_BVQA_model_L1RankLoss_resize_520_crop_size_448_exp_version_0.log

 CUDA_VISIBLE_DEVICES="2,3" python -u train_ugc.py \
 --database youtube_ugc \
 --model_name UGC_BVQA_model \
 --conv_base_lr 0.00001 \
 --epochs 100 \
 --train_batch_size 2 \
 --print_samples 1000 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --exp_version 0 \
 --loss_type L1RankLoss \
 --resize 520 \
 --crop_size 448 \
 >> logs/train_UGC_BVQA_model_L1RankLoss_resize_520_crop_size_448_exp_version_0.log