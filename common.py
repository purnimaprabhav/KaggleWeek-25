import time
from functools import wraps
from typing import List, Dict
import pandas as pd


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def parse_input(filename: str) -> pd.DataFrame:
    rows = []
    with open(filename, 'r', encoding='utf-8') as f:
        n = int(f.readline().strip())
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            orientation, tag_count, rest = line.strip().split(maxsplit=2)
            tags = rest.split()
            rows.append({
                'index': idx,
                'type': orientation,
                'tag_count': int(tag_count),
                'tags': tags
            })
    return pd.DataFrame(rows)


def build_frameglasses(df: pd.DataFrame) -> List[Dict]:
    landscapes = df[df['type'] == 'L']
    portraits = df[df['type'] == 'P']
    frameglasses = []

    # Single landscapes
    for _, row in landscapes.iterrows():
        frameglasses.append({'indices': [row['index']], 'tags': set(row['tags'])})

    # Pair portraits sequentially
    pf_indices = portraits['index'].tolist()
    for i in range(0, len(pf_indices) - 1, 2):
        i1, i2 = pf_indices[i], pf_indices[i + 1]
        tags1 = set(df.loc[df['index'] == i1, 'tags'].values[0])
        tags2 = set(df.loc[df['index'] == i2, 'tags'].values[0])
        frameglasses.append({'indices': [i1, i2], 'tags': tags1 | tags2})

    return frameglasses


def write_output(frameglasses: List[Dict], output_filename: str):
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(frameglasses)}\n")
        for fg in frameglasses:
            if len(fg['indices']) == 1:
                f.write(f"{fg['indices'][0]}\n")
            else:
                f.write(" ".join(map(str, fg['indices'])) + "\n")


def local_satisfaction_score(tags_a: set, tags_b: set) -> int:
    common = tags_a & tags_b
    only_a = tags_a - tags_b
    only_b = tags_b - tags_a
    return min(len(common), len(only_a), len(only_b))


@timeit
def global_satisfaction_score(frameglasses: List[Dict]) -> int:
    score = 0
    for i in range(len(frameglasses) - 1):
        tags_i = frameglasses[i]['tags']
        tags_j = frameglasses[i + 1]['tags']
        score += local_satisfaction_score(tags_i, tags_j)
    return score
