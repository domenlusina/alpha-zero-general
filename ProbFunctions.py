def cutoff(move, start_prob=0.9, cutoffmove=10):
    if move > cutoffmove:
        return 0
    return start_prob


def linearf(curIter, totalIter, start_prob, end_prob):
    frac = (curIter + 1) / totalIter
    return max(0, min(1, (1 - frac) * start_prob + frac * end_prob))

