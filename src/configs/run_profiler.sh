uv run nsys profile -o results python3 src/profile_benchmarks/profile_fwd_and_bwd_passes.py \
    --model_config src/configs/sft_train_owt_contrastive.json \
    --batch-size 4 \
    --context-length 512 \
    --device cuda \
    --num-warmup-steps 5
