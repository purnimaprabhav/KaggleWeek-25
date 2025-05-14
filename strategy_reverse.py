
import argparse
import common

def main():
    parser = argparse.ArgumentParser(description="Reverse-order strategy")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    frames = common.parse_input(args.input_file)
    ordered, t_order = common.time_function(common.order_reverse, frames)
    score, t_score = common.time_function(common.compute_score, ordered)
    common.write_output(args.output_file, ordered)

    print(f"Strategy: Reverse Order")
    print(f"Ordering time: {t_order:.4f}s")
    print(f"Scoring time:  {t_score:.4f}s")
    print(f"Score:         {score}")

if __name__ == '__main__':
    main()
