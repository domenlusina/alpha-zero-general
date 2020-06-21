import random


def opening_tree():
    d2moves = [[3, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 4, 3, 6, 4, 1, 1, 1, 1, 5, 5, 5, 5],
               [3, 3, 3, 3, 3, 4, 4, 4, 4, 1, 4, 1, 1, 0, 0, 0, 0],
               [3, 3, 3, 3, 3, 4, 1, 4, 0, 2, 2, 2, 2, 1, 2],
               [3, 3, 3, 3, 3, 4, 1, 0, 4, 1, 4, 4, 0, 1, 1, 1, 3, 0, 4],
               [3, 3, 3, 3, 3, 4, 1, 0, 1],
               [3, 3, 3, 3, 3, 4, 1, 0, 4, 1, 0, 1, 1],
               [3, 3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 1, 5],
               [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 5, 4, 4, 3, 4],
               [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 5, 4, 4, 4, 6, 3, 4, 5, 5, 6, 2, 5, 5],
               [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 5, 4, 4, 4, 3],
               [3, 3, 3, 3, 3, 4],
               [3, 3, 3, 3, 3, 2, 1, 5, 4, 4, 4, 4, 1, 2, 4, 2, 2, 2, 6, 5, 1, 1, 1, 5],
               [3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 4, 4, 2, 0, 6, 2, 6, 6, 4, 4, 4, 0, 0, 0, 0, 0, 4, 1, 1],
               [3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 4, 4, 2, 0, 6, 2, 6, 6, 4, 4, 4, 0, 4],
               [3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 4, 4, 2, 0, 6, 2, 6, 6, 4, 4, 4, 0, 0, 0, 4],
               [3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 4, 4, 2, 0, 6, 2, 4],
               [3, 3, 3, 2, 2, 2, 3, 3, 6, 2, 1, 2, 2, 4, 4],
               [3, 3, 3, 2, 2, 2, 3, 3, 6, 4, 6, 6, 4, 6],
               [3, 3, 3, 5],
               [3, 3, 3, 6],
               [3, 3, 3, 3]]

    b1moves = [[3, 1, 5, 4, 5, 5, 3, 3, 2, 3, 5],
               [3, 1, 1],
               [3, 1, 5, 4, 5, 5, 3, 3, 3],
               [3, 1, 5, 4, 5, 5, 3, 3, 5]]

    e1moves = [[3, 4, 0, 3, 3, 3, 1, 2, 2, 2, 4, 2, 3, 2, 2, 5, 3],
               [3, 4, 0, 3, 1],
               [3, 4, 0, 3, 3, 3, 1, 2, 2, 2, 2],
               [3, 4, 0, 3, 3, 3, 1, 2, 2, 2, 3],
               [3, 4, 0, 2, 2, 2, 2, 4, 4, 3, 4, 4, 5, 3, 3, 3, 5],
               [3, 4, 0, 2, 2, 2, 2, 4, 4, 3, 4, 4, 3],
               [3, 4, 0, 2, 2, 2, 2, 4, 4, 4, 4],
               [3, 4, 0, 2, 2, 2, 2, 4, 4, 4, 3, 3, 4, 2, 0],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 3, 3],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 3, 3, 3, 3, 3, 1],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 1, 3, 3, 4],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 1, 3, 3, 3],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 1, 3, 3, 0],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 1, 3, 3, 0, 3, 0],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 1, 3, 3, 3],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 3, 1, 6, 6, 2, 3],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 3, 3],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 3, 1, 6, 6, 2, 2],
               [3, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 4, 0, 0, 4, 1, 3, 3, 3],
               [3, 4, 1, 0, 1, 1, 3],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 1, 0, 3, 0, 0, 0, 4],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 1, 0, 3, 0, 0, 0, 2, 2, 4, 1, 6],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 1, 0, 3, 0, 0, 0, 6],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 1, 0, 6],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 1, 0, 4],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 6],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 3, 3, 3],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 1],
               [3, 4, 1, 4, 1, 1, 3, 4, 4, 0, 4],
               [3, 4, 1, 4, 1, 1, 0],
               [3, 4, 1, 4, 1, 1, 2],
               [3, 4, 1, 4, 1, 1, 4],
               [3, 4, 1, 4, 4],
               [3, 4, 1, 4, 0, 2, 1, 1, 2, 2, 3, 0, 3, 3],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 0, 0, 1, 1, 3, 3],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 0, 0, 1, 1, 3, 4],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 0, 4],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 6, 6, 1, 1, 4, 4],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 2, 0],
               [3, 4, 1, 1, 0, 2, 2, 2, 2, 4, 2, 2, 4],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 2, 3, 3, 3, 3, 3, 6],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 2, 3, 3, 3, 6, 3, 0, 5, 0, 0, 6, 5, 6],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 2, 3, 3, 3, 6, 3, 0, 5, 6],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 6, 2, 2, 0, 6],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 6, 2, 2, 0, 4, 4, 4, 4, 6, 3, 6, 6, 3],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 6, 2, 2, 0, 4, 4, 4, 4, 0],
               [3, 4, 1, 1, 0, 2, 2, 2, 4, 2, 6, 2, 2, 0, 4, 4, 4, 4, 6, 3, 3],
               [3, 4, 1, 1, 0, 2, 4, 1, 1, 0, 2],
               [3, 4, 1, 1, 0, 2, 4, 1, 1, 2, 0],
               [3, 4, 1, 1, 0, 2, 4, 1, 1, 3, 3],
               [3, 4, 1, 1, 0, 2, 4, 1, 1, 4, 1],
               [3, 4, 1, 1, 0, 2, 4, 2, 0, 1, 1]]

    openings = d2moves
    return random.choice(openings)
