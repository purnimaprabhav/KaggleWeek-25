import time
import random
from itertools import chain

# GPU support via PyTorch
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_GPU = False

class FrameGlass:
    __slots__ = ('ids', 'tags')
    def __init__(self, ids, tags):
        # ids: tuple of photo indices (single or paired)
        self.ids = tuple(ids)
        self.tags = tags


def parse_input(path):
    """
    Reads the input file of photos and returns two lists:
    - landscapes: list of (id, tags) for 'L' rows
    - portraits: list of (id, tags) for 'P' rows

    File format:
      N
      orientation tag_count tag1 tag2 ...
    orientation: 'L' for landscapes, 'P' for portraits
    """
    landscapes = []
    portraits = []
    with open(path, 'r') as f:
        _ = int(f.readline().strip())
        for idx, line in enumerate(f):
            parts = line.strip().split()
            ori = parts[0]
            tcount = int(parts[1])
            tags = set(parts[2:2 + tcount])
            if ori == 'L':
                landscapes.append((idx, tags))
            elif ori == 'P':
                portraits.append((idx, tags))
            # ignore other orientations
    return landscapes, portraits


def create_slides(landscapes, portraits, pairing_strategy=None, seed=None):
    """
    Generates FrameGlass slides:
      - Each landscape becomes its own slide.
      - Portraits are paired (default: smallest with largest tag sets).
    pairing_strategy: optional function to pair portrait items.
    """
    slides = [FrameGlass([pid], tags) for pid, tags in landscapes]
    verts = portraits[:]
    if pairing_strategy:
        pairs = pairing_strategy(verts)
    else:
        verts_sorted = sorted(verts, key=lambda x: len(x[1]))
        pairs = []
        i, j = 0, len(verts_sorted) - 1
        while i < j:
            pairs.append((verts_sorted[i], verts_sorted[j]))
            i += 1
            j -= 1
    for (id1, t1), (id2, t2) in pairs:
        slides.append(FrameGlass([id1, id2], t1 | t2))
    return slides


def write_output(path, slides):
    """
    Writes slideshow to file:
      First line: number of slides
      Following lines: space-separated photo ids per slide
    """
    with open(path, 'w') as f:
        f.write(f"{len(slides)}\n")
        for slide in slides:
            f.write(' '.join(str(i) for i in slide.ids) + "\n")


def order_greedy_sample(slides, sample_size=100, seed=None):
    if seed is not None:
        random.seed(seed)
    remaining = slides[:]
    random.shuffle(remaining)
    current = remaining.pop()
    ordered = [current]
    while remaining:
        m = min(sample_size, len(remaining))
        idxs = random.sample(range(len(remaining)), m)
        best_idx, best_score = idxs[0], -1
        for idx in idxs:
            cand = remaining[idx]
            common = len(current.tags & cand.tags)
            only_cur = len(current.tags - cand.tags)
            only_cand = len(cand.tags - current.tags)
            sc = min(common, only_cur, only_cand)
            if sc > best_score:
                best_score, best_idx = sc, idx
        current = remaining.pop(best_idx)
        ordered.append(current)
    return ordered


import torch
import random
from itertools import chain

def order_greedy_gpu(slides, sample_size=None, seed=None):
    """
    Hybrid GPU-accelerated greedy selection with chunked scoring:
    - Constructs tag matrix on CPU to avoid CUDA OOM
    - Moves only small chunks to GPU for scoring
    - Supports slide sampling and reproducible randomness

    Args:
        slides (list): List of slide objects with `.tags` attribute
        sample_size (int, optional): Number of slides to sample from `slides`. Defaults to None (use all).
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    assert torch.cuda.is_available(), "GPU unavailable or torch not installed"

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if sample_size is not None and sample_size < len(slides):
        slides = random.sample(slides, sample_size)

    # Build tag index mapping
    tag_set = set(chain.from_iterable(s.tags for s in slides))
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}
    num_tags = len(tag2idx)
    num_slides = len(slides)

    # Construct binary matrix [num_slides x num_tags] on CPU
    mat = torch.zeros((num_slides, num_tags), dtype=torch.bool, device='cpu')
    for i, s in enumerate(slides):
        idxs = [tag2idx[t] for t in s.tags]
        mat[i, idxs] = True

    remaining = list(range(num_slides))

    # Random initial pick
    first = random.choice(remaining)
    ordered_idx = [first]
    remaining.remove(first)

    # Greedy selection loop
    batch_size = 1024  # Adjust based on GPU memory
    while remaining:
        cur_vec = mat[ordered_idx[-1]].unsqueeze(0).to('cuda')  # [1 x num_tags]
        scores = []

        for i in range(0, len(remaining), batch_size):
            batch_idxs = remaining[i:i + batch_size]
            rem_mat = mat[batch_idxs].to('cuda')  # [B x num_tags]

            common = (cur_vec & rem_mat).sum(dim=1)
            only_cur = (cur_vec & ~rem_mat).sum(dim=1)
            only_rem = (~cur_vec & rem_mat).sum(dim=1)
            score = torch.min(torch.min(common, only_cur), only_rem)

            scores.append(score.cpu())  # Move back to CPU

        all_scores = torch.cat(scores)  # [len(remaining)]
        best_pos = torch.argmax(all_scores).item()
        best_slide = remaining.pop(best_pos)
        ordered_idx.append(best_slide)

    return [slides[i] for i in ordered_idx]


"""
def order_greedy_gpu(slides):
    assert HAS_GPU, "GPU unavailable or torch not installed"
    # Build tag index mapping
    tag_set = set(chain.from_iterable(s.tags for s in slides))
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}
    num_tags = len(tag2idx)
    num_slides = len(slides)
    # Construct binary matrix [num_slides x num_tags] on GPU
    mat = torch.zeros((num_slides, num_tags), dtype=torch.bool, device='cuda')
    for i, s in enumerate(slides):
        idxs = [tag2idx[t] for t in s.tags]
        mat[i, idxs] = True
    remaining = list(range(num_slides))
    # Random initial pick
    first = random.choice(remaining)
    ordered_idx = [first]
    remaining.remove(first)
    # Greedy selection
    while remaining:
        cur_vec = mat[ordered_idx[-1]].unsqueeze(0)  # [1 x num_tags]
        rem_idxs = torch.tensor(remaining, device='cuda')
        rem_mat = mat[rem_idxs]  # [R x num_tags]
        common = (cur_vec & rem_mat).sum(dim=1)
        only_cur = (cur_vec & ~rem_mat).sum(dim=1)
        only_rem = (~cur_vec & rem_mat).sum(dim=1)
        scores = torch.min(torch.min(common, only_cur), only_rem)
        best_pos = torch.argmax(scores).item()
        best_slide = remaining.pop(best_pos)
        ordered_idx.append(best_slide)
    return [slides[i] for i in ordered_idx]
"""

def compute_score(slides):
    """
    Computes total interest score for a slideshow.
    """
    score = 0
    for a, b in zip(slides, slides[1:]):
        common = a.tags & b.tags
        only_a = a.tags - b.tags
        only_b = b.tags - a.tags
        score += min(len(common), len(only_a), len(only_b))
    return score


def time_function(fn, *args, **kwargs):
    """
    Times function execution: returns (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def random_pairing(portrait_list, seed=None):
    """
    Randomly pair portrait items.
    """
    if seed is not None:
        random.seed(seed)
    v = portrait_list[:]
    random.shuffle(v)
    return [(v[i], v[i+1]) for i in range(0, len(v)-1, 2)]

def best_pairing(portrait_list, sample_size=50, seed=None):
    """
    Greedy best pairing: repeatedly take one portrait and match it
    to the candidate (among a small random sample) with the fewest shared tags,
    so each slide covers the largest possible tag set.
    """
    if seed is not None:
        random.seed(seed)
    rem = portrait_list[:]          # [(id, tags), …]
    pairs = []
    while len(rem) > 1:
        pid, tags = rem.pop()        # take “current” portrait
        m = min(sample_size, len(rem))
        # sample a few candidates rather than scanning all remaining
        idxs = random.sample(range(len(rem)), m)
        best_idx, best_overlap = idxs[0], float('inf')
        for i in idxs:
            other_tags = rem[i][1]
            overlap = len(tags & other_tags)
            if overlap < best_overlap:
                best_overlap, best_idx = overlap, i
        other = rem.pop(best_idx)
        pairs.append(((pid, tags), other))
    return pairs
