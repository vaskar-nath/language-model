uv run python3 src/tokenizer.py \
    --tokenizer_path tokenizers/TinyStoriesV2-GPT4-train_tokenizer \
    --input_path data/TinyStoriesV2-GPT4-train.txt \
    --output_path data/TinyStoriesV2-GPT4-train_token_ids.npy \
    --show_progress

uv run python3 src/tokenizer.py \
    --tokenizer_path tokenizers/TinyStoriesV2-GPT4-train_tokenizer \
    --input_path data/TinyStoriesV2-GPT4-valid.txt \
    --output_path data/TinyStoriesV2-GPT4-valid_token_ids.npy \
    --show_progress