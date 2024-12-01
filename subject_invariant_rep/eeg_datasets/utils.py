import itertools

import numpy as np


def extract_combinations(labels, repeat):
    def contains_duplicates(X):
        return len(np.unique(X)) != len(X)

    out = list(itertools.product(labels, repeat=repeat))
    out = filter(lambda x: not contains_duplicates(x), out)
    # out = list(k for k, _ in itertools.groupby(out))
    # Convert inner lists to frozensets and store them in a set to keep track of unique sets
    unique_frozensets = set(frozenset(inner_list) for inner_list in out)

    # Convert unique frozensets back to lists
    unique_lists = [list(frozenset_) for frozenset_ in unique_frozensets]
    out = unique_lists
    # out = list(out)
    out = [sorted(inner_list) for inner_list in out]
    out.sort()
    return out


def create_class_combinations(labels):
    out = []
    for i in range(2, len(labels) + 1):
        out += extract_combinations(labels, i)
    out.sort()
    return out


__all__ = ["create_class_combinations"]
