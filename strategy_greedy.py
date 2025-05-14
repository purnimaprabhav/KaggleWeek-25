import argparse
import common

def main():
    parser = argparse.ArgumentParser(
        description='Greedy sampling strategy for maximal interest score'
    )
    parser.add_argument(
        'input_file', help='Photo dataset input file'
    )
    parser.add_argument(
        'output_file', help='Slide show output file'
    )
    parser.add_argument(
        '--pair_strategy', choices=['default', 'random', 'best'], default='default',
        help='Strategy for pairing portraits'
    )

    parser.add_argument(
        '--sample_size', type=int, default=100,
        help='Candidates to sample at each greedy step'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    landscapes, portraits = common.parse_input(args.input_file)
    if args.pair_strategy == 'random':
        pair_func = lambda v: common.random_pairing(v, seed=args.seed)
    elif args.pair_strategy == 'best':
        pair_func = lambda v: common.best_pairing(v, sample_size=100, seed=args.seed)
    else:
        pair_func = None


    slides, t_pair = common.time_function(
        common.create_slides,
        landscapes, portraits, pair_func, args.seed
    )

    ordered, t_order = common.time_function(
        common.order_greedy_sample,
        slides, args.sample_size, args.seed
    )
    score, t_score = common.time_function(common.compute_score, ordered)

    common.write_output(args.output_file, ordered)

    print(f"Pairing time:       {t_pair:.4f}s")
    print(f"Ordering time:      {t_order:.4f}s")
    print(f"Scoring time:       {t_score:.4f}s")
    print(f"Slides count:       {len(ordered)}")
    print(f"Total score:        {score}")

if __name__ == '__main__':
    main()
