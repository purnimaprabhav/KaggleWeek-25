import argparse
import common
import torch

def main():
    parser = argparse.ArgumentParser(
        description='Greedy GPU strategy with pairing options'
    )
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument(
        '--pair_strategy',
        choices=['default', 'random', 'best'],
        default='default',
        help='Portrait pairing: default, random, or best'
    )
    parser.add_argument(
        '--sample_size', type=int, default=50,
        help='Sample size for best_pairing'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
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
    ordered, t_order = common.time_function(common.order_greedy_gpu, slides, sample_size=args.sample_size, seed=args.seed)

    score, t_score = common.time_function(common.compute_score, ordered)
    common.write_output(args.output_file, ordered)

    print(f"Pairing time:       {t_pair:.4f}s")
    print(f"GPU Ordering time:  {t_order:.4f}s")
    print(f"Scoring time:       {t_score:.4f}s")
    print(f"Slides:             {len(ordered)}")
    print(f"Score:              {score}")

if __name__ == '__main__':
    main()