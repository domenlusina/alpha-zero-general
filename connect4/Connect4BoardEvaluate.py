import os
from subprocess import run, PIPE

import numpy as np


def getBoardScore(moves):
    if os.name == 'nt':
        path = "C:\\Magistrsko_delo\\connect4\\bin\\board_evaluate.exe"
    else:
        path = "/home/dlusina/connect4/bin/board_evaluate"
    if not isinstance(moves, str):
        moves = "".join([str(move + 1) for move in moves])

    process = run(path, stdout=PIPE, input=(moves + "\n").encode())
    score = process.returncode

    if score > 42:
        score = np.uint32(score).view('int32')

    if score > 0:
        score = 1
    elif score < 0:
        score = -1

    return score


def changeInTheoreticalGameValues(moves):
    # a function that returns which move (starting with 0) changed the theoretical value of first player
    # (first player did not play perfect and made a mistake which made him not win (he either drew or lost))
    gameValues = [getBoardScore(moves[:i + 1]) for i in range(len(moves))]

    expectedValue = -1
    for i, gameValue in enumerate(gameValues):
        if expectedValue != gameValue:
            return i

        expectedValue *= -1
    return None


def analyseEngine(path):
    """
    We open a file containing moves that were played when playing agains engine player.
    Looking at which moves AlphaZero made when playing as first player we return the average number of moves until
    the game theoretical value was changed.

    :param path: path to engine.txt file
    :return: average number of moves until game theoretical value changes
    """

    movesList = []

    avg_scores = []

    with open(path, 'r') as f:
        going_second = False
        for line in f:
            if not going_second:
                if line[0] == "#":
                    going_second = True
                elif line[0] == "\t":
                    movesList.append(line.split()[0])
            if going_second and (line[0] == "(" or line[0] == "["):
                going_second = False

                scores = [changeInTheoreticalGameValues(moves) for moves in movesList]
                scores = [42 if score is None else score for score in scores]
                print(scores)

                avg_scores.append(sum(scores) / len(scores))
                movesList = []

    scores = [changeInTheoreticalGameValues(moves) for moves in movesList]
    scores = [42 if score is None else score for score in scores]
    print(scores)

    avg_scores.append(sum(scores) / len(scores))

    return avg_scores

# print(changeInTheoreticalGameValues("444441566623"))
