import argparse
import common
import torch
import os
import sys

jobs = [
    ['Data/0_example.txt',      'output/output_0_example.txt', '--pair_strategy', 'random', '--sample_size', '3', '--seed', '45'],
    ['Data/10_computable_moments.txt', 'output/output_10_computable_moments.txt', '--pair_strategy', 'best', '--sample_size', '4000', '--seed', '48'],
    ['Data/11_randomizing_paintings.txt', 'output/output_11_randomizing_paintings.txt', '--pair_strategy', 'best', '--sample_size', '1', '--seed', '44'],
    ['Data/110_oily_portraits.txt', 'output/output_110_oily_portraits.txt', '--pair_strategy', 'best', '--sample_size', '200', '--seed', '58']
]

def run_job(args_list):
    sys.argv = ['run1_all(gpu).py'] + args_list  # Simulate CLI args

    current_path = os.getcwd()
    print("Current working directory:", current_path)

    parser = argparse.ArgumentParser(
        description='Greedy GPU strategy with pairing options'
    )
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--pair_strategy', choices=['default', 'random', 'best'], default='default',
                        help='Portrait pairing: default, random, or best')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Sample size for best_pairing')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    args = parser.parse_args()

    # Env check
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('GPU enabled in common:', common.HAS_GPU)
    if not common.HAS_GPU:
        raise RuntimeError('GPU support not available')

    landscapes, portraits = common.parse_input(args.input_file)
    if args.pair_strategy == 'random':
        pair = lambda v: common.random_pairing(v, seed=args.seed)
    elif args.pair_strategy == 'best':
        pair = lambda v: common.best_pairing(v, sample_size=args.sample_size, seed=args.seed)
    else:
        pair = None

    slides, t_pair = common.time_function(
        common.create_slides,
        landscapes, portraits, pair, args.seed
    )
    ordered, t_order = common.time_function(common.order_greedy_gpu, slides)

    score, t_score = common.time_function(common.compute_score, ordered)
    common.write_output(args.output_file, ordered)

    print(f"Processed: {args.input_file} → {args.output_file}")
    print(f"Pairing time:       {t_pair:.4f}s")
    print(f"GPU Ordering time:  {t_order:.4f}s")
    print(f"Scoring time:       {t_score:.4f}s")
    print(f"Slides:             {len(ordered)}")
    print(f"Score:              {score}")
    print('-' * 60)


def main():
    current_path = os.getcwd()
    print("Current working directory:", current_path)
    for job_args in jobs:
        run_job(job_args)

if __name__ == '__main__':
    main()
