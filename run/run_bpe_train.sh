uv run python3 src/train_bpe.py \
    --input_path data/TinyStoriesV2-GPT4-train.txt\
    --output_dir tokenizers/TinyStoriesV2-GPT4-train_tokenizer \
    --vocab_size 10000 \
    --special_tokens "<|endoftext|>" "<|pad|>" \
    --verbose