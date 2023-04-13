CUDA_VISIBLE_DEVICES=0 python -u test_demo.py \
--method_name single-scale \
--dist videos/2999049224_original_centercrop_960x540_8s.mp4 \
--output result.txt \
--is_gpu \
>> logs/test_demo.log

CUDA_VISIBLE_DEVICES=2,3 python -u test_on_pretrained_model.py \
--database KoNViD-1k \
--train_database LSVQ \
--model_name UGC_BVQA_model \
--feature_type SlowFast \
--trained_model UGC_BVQA_model.pth \
--num_workers 6 \