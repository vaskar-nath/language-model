# CUDA_VISIBLE_DEVICES=1 uv run python3 /mnt/efs/vaskarnath/practice/language-model/src/trainer.py \
#     --config /mnt/efs/vaskarnath/practice/language-model/src/configs/sft_train_owt.json

# CUDA_VISIBLE_DEVICES=1 uv run python3 /mnt/efs/vaskarnath/practice/language-model/src/trainer.py \
#     --config /mnt/efs/vaskarnath/practice/language-model/src/configs/sft_train_tiny_stories.json

CUDA_VISIBLE_DEVICES=4 uv run python3 /mnt/efs/vaskarnath/practice/language-model/src/trainer.py \
    --config /mnt/efs/vaskarnath/practice/language-model/src/configs/sft_train_owt_contrastive.json

