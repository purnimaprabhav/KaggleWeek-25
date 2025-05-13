import time
import random
from typing import List, Dict, Set

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] executed in {elapsed:.4f} seconds")
        return result
    return wrapper

@timeit
def parse_input(file_path: str) -> List[Dict]:
    paintings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        n = int(f.readline())
        for idx in range(n):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            orientation = parts[0]
            tag_count = int(parts[1])
            tags = set(parts[2:])
            paintings.append({
                'id': idx,
                'type': orientation,
                'tags': tags
            })
    return paintings

@timeit
def build_frameglasses(paintings: List[Dict]) -> List[Dict]:
    landscapes = [p for p in paintings if p['type'] == 'L']
    portraits = [p for p in paintings if p['type'] == 'P']

    frameglasses = []

    # Process landscapes
    for p in landscapes:
        frameglasses.append({
            'ids': [p['id']],
            'tags': p['tags'].copy()
        })

    # Pair portraits
    i = 0
    while i + 1 < len(portraits):
        p1 = portraits[i]
        p2 = portraits[i + 1]
        combined_tags = p1['tags'].union(p2['tags'])
        frameglasses.append({
            'ids': [p1['id'], p2['id']],
            'tags': combined_tags
        })
        i += 2

    # Handle odd portrait
    if i < len(portraits):
        p = portraits[i]
        frameglasses.append({
            'ids': [p['id']],
            'tags': p['tags'].copy()
        })

    return frameglasses

def local_satisfaction_score(tags_a: Set[str], tags_b: Set[str]) -> int:
    common = len(tags_a & tags_b)
    only_a = len(tags_a - tags_b)
    only_b = len(tags_b - tags_a)
    return min(common, only_a, only_b)

def global_satisfaction_score(frames_order: List[Dict]) -> int:
    score = 0
    for i in range(len(frames_order) - 1):
        score += local_satisfaction_score(frames_order[i]['tags'], frames_order[i + 1]['tags'])
    return score

def order_same(frames: List[Dict]) -> List[Dict]:
    return frames.copy()

def order_reversed(frames: List[Dict]) -> List[Dict]:
    return frames[::-1]

def order_random(frames: List[Dict]) -> List[Dict]:
    shuffled = frames.copy()
    random.shuffle(shuffled)
    return shuffled

def order_by_tag_count(frames: List[Dict]) -> List[Dict]:
    return sorted(frames, key=lambda x: len(x['tags']))

@timeit
def run_strategies(frameglasses: List[Dict]) -> List[Dict]:
    strategies = {
        "original_order": order_same,
        "reversed_order": order_reversed,
        "random_order": order_random,
        "sorted_by_tag_count": order_by_tag_count
    }

    results = []
    
    for name, strategy in strategies.items():
        ordered = strategy(frameglasses)
        score = global_satisfaction_score(ordered)
        results.append({
            'strategy': name,
            'order': ordered,
            'score': score
        })
        print(f"Strategy: {name:<20} | Score: {score}")

    best_result = max(results, key=lambda x: x['score'])
    print(f"\nBest strategy: {best_result['strategy']} with score {best_result['score']}")
    return best_result['order']

def write_output(frames_order: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(frames_order)}\n")
        for frame in frames_order:
            f.write(' '.join(map(str, frame['ids'])) + '\n')

def main():
    input_file = input("Enter the path to your input file (.txt): ").strip()
    output_file = input("Enter the desired output file name (e.g., output.txt): ").strip()

    print("\nProcessing input file...")
    paintings = parse_input(input_file)
    print(f"Found {len(paintings)} paintings")

    print("\nBuilding frameglasses...")
    frameglasses = build_frameglasses(paintings)
    print(f"Created {len(frameglasses)} frameglasses")

    print("\nEvaluating different ordering strategies...")
    best_order = run_strategies(frameglasses)

    print(f"\nWriting best solution to {output_file}...")
    write_output(best_order, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
