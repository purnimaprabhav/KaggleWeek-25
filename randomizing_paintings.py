import time
import random
from pathlib import Path

def parse_input_file(filepath):
    """Reads the painting data from the given file."""
    painting_list = []
    with open(filepath, 'r') as file:
        total_paintings = int(file.readline().strip())
        for index in range(total_paintings):
            parts = file.readline().strip().split()
            orientation, tag_count = parts[0], int(parts[1])
            tags = set(parts[2:])
            painting_list.append((index, orientation, tags))
    return painting_list

def construct_frameglasses(painting_list):
    """Converts paintings into frameglasses."""
    results = []
    landscape = [(idx, tags) for idx, orient, tags in painting_list if orient == 'L']
    portrait = [(idx, tags) for idx, orient, tags in painting_list if orient == 'P']

    # Process landscape directly
    results.extend(([idx], tags) for idx, tags in landscape)

    # Pair portrait images
    for i in range(0, len(portrait) - 1, 2):
        idx1, tags1 = portrait[i]
        idx2, tags2 = portrait[i + 1]
        results.append(([idx1, idx2], tags1 | tags2))

    return results

def strategy_identity(fgs):
    return fgs

def strategy_reversed(fgs):
    return list(reversed(fgs))

def strategy_shuffle(fgs):
    shuffled = fgs[:]
    random.shuffle(shuffled)
    return shuffled

def strategy_tagcount(fgs):
    return sorted(fgs, key=lambda fg: len(fg[1]))

def output_results(ordered_fgs, filepath):
    with open(filepath, 'w') as out_file:
        out_file.write(f"{len(ordered_fgs)}\n")
        for ids, _ in ordered_fgs:
            out_file.write(" ".join(map(str, ids)) + "\n")

def compute_local_score(fg1, fg2):
    tags_a, tags_b = fg1[1], fg2[1]
    return min(len(tags_a & tags_b), len(tags_a - tags_b), len(tags_b - tags_a))

def compute_total_score(ordered_fgs):
    return sum(compute_local_score(ordered_fgs[i], ordered_fgs[i + 1]) for i in range(len(ordered_fgs) - 1))

def run_strategies(input_path, output_folder, team_id):
    input_path = Path(input_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start = time.time()
    paintings = parse_input_file(input_path)
    time_parsing = time.time() - start

    start = time.time()
    frameglasses = construct_frameglasses(paintings)
    time_construction = time.time() - start

    strategies = [
        ("original", strategy_identity),
        ("reversed", strategy_reversed),
        ("shuffled", strategy_shuffle),
        ("tag_sorted", strategy_tagcount),
    ]

    results = []

    for name, strategy in strategies:
        t1 = time.time()
        ordered = strategy(frameglasses)
        t2 = time.time()
        score = compute_total_score(ordered)
        t3 = time.time()
        output_file = output_folder / f"Team_{team_id}_{name}.txt"
        output_results(ordered, output_file)
        t4 = time.time()

        results.append({
            "strategy": name,
            "score": score,
            "total_time": (t4 - start),
            "output": output_file
        })

        print(f"Strategy: {name} | Score: {score} | Time: {t4 - start:.4f}s | File: {output_file}")

    return results

def main():
    team_id = "18"
    
    # Get input file path directly from user
    input_file = input("Enter the path to your input file: ").strip()
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return
    
    # Create output folder based on input file name
    output_folder = Path("Outputs") / input_path.stem
    
    print("\nProcessing file:", input_path)
    print("Output will be saved to:", output_folder)
    
    results = run_strategies(input_path, output_folder, team_id)

    print("\n--- Summary ---")
    for res in results:
        print(f"{res['strategy'].capitalize()}: Score = {res['score']}")

if _name_ == "_main_":
    main()
