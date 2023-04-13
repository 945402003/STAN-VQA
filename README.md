# STAN-VQA
STAN: Spatio-Temporal Alignment Network for No-Reference Video Quality Assessment

### Install Requirements
```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

### Download databases
[LSVQ](https://github.com/baidut/PatchVQ)
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)
[Youtube-UGC](https://media.withyoutube.com/)

### Train models
1. Extract video frames
```shell
python -u extract_frame_LSVQ.py >> logs/extract_frame_LSVQ.log
```
2. Extract motion features
```shell
 CUDA_VISIBLE_DEVICES=0 python -u extract_SlowFast_features_LSVQ.py \
 --database LSVQ \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder LSVQ_SlowFast_feature/ \
 >> logs/extracted_LSVQ_SlowFast_features.log
```
3. Train the model
```shell
 CUDA_VISIBLE_DEVICES=0 python -u train_baseline.py \
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
```

Test on the public VQA database
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_on_pretrained_model.py \
--database KoNViD-1k \
--train_database LSVQ \
--model_name UGC_BVQA_model \
--feature_type SlowFast \
--trained_model ckpts/UGC_BVQA_model.pth \
--num_workers 6 \
>> logs/test_on_KoNViD-1k_train_on_LSVQ.log
```

Test on a single video
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_demo.py \
--method_name single-scale \
--dist videos/2999049224_original_centercrop_960x540_8s.mp4 \
--output result.txt \
--is_gpu \
>> logs/test_demo.log
```

