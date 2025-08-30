import argparse
import json
import torch
import timeit


from src.language_model.transformer_lm import TransformerLM
from src.train.cross_entropy import cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile forward and backward passes of a model."
    )
    parser.add_argument(
        "--model_config", type=str, required=True,
        help="Path for the model configs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for profiling (default: 32)."
    )
    parser.add_argument(
        "--context-length", type=int, default=1024,
        help="Batch size for profiling (default: 32)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run the profiling on (default: cuda)."
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=5,
        help="Number of warm up steps before profiling (default: 5)."
    )
    return parser.parse_args()

def profile_time(func, args, desc):
    start_time = timeit.default_timer()
    output = func(**args)
    end_time = timeit.default_timer()
    print(f"{desc}: {end_time - start_time:.4f} seconds")   
    return output

def main(args):
    config = json.load(open(args.model_config))
    model = TransformerLM(
            **config['model_configs'],
            ).to(args.device)
    
    batch_size = args.batch_size
    context_length = args.context_length
    vocab_size = config['model_configs']['vocab_size']

    random_data = torch.randint(0, vocab_size-1, (batch_size, context_length + 1), device=args.device)
    input_data = random_data[:, :-1]
    labels = random_data[:, 1:]

    for _ in range(args.num_warmup_steps):
        logits = model(input_data)
        loss = cross_entropy(logits, labels)
        loss.backward()

    logits = profile_time(model, {'x': input_data}, "fwd pass")
    loss = cross_entropy(logits, labels)
    profile_time(loss.backward, {}, "bwd_pass")


if __name__ == "__main__":
    args = parse_args()
    main(args)