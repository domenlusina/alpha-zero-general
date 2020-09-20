import ast
import os
from collections import Counter
from subprocess import run, PIPE

import numpy as np

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import EngineConnect4Player
from tests.test_nn import get_board

from connect4.Connect4Players import *
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict
from MCTS import MCTS


def getMoveScores(moves):
    if not isinstance(moves, str):
        moves = "".join([str(move + 1) for move in moves])

    c = Counter(moves)
    scores = []
    for i in range(1, 8):
        if c[str(i)] < 6:
            score = getBoardScore(moves + str(i))
            scores.append(-score)

    return scores


def evaluateMove(moves):
    """
    Evaluates the last move in moves. We create a unique scores of legal moves with consideration that the state of the
    board is obtained  by preforming all the moves except the last one.

    :param moves: moves that have been preformed given either as a string with char 1-7 or a list with values 0-6
    :return:
    """
    if not isinstance(moves, str):
        moves = "".join([str(move + 1) for move in moves])

    move = moves[-1]
    moves = moves[:-1]

    c = Counter(moves)
    move_score = -1

    scores = []
    for i in range(1, 8):
        if c[str(i)] < 6:
            score = -getBoardScore(moves + str(i))
            scores.append(score)
            if move == str(i):
                move_score = score

    scores = sorted(set(scores), reverse=True)

    return scores.index(move_score), len(scores)


def goodMoveProb(moves):
    """
    For a given state of the board described with moves, we return how many of the moves we can preform are
    good (winning) and how many legal moves are there.

    :param moves: moves that have been preformed given either as a string with char 1-7 or a list with values 0-6
    :return:
    """
    if not isinstance(moves, str):
        moves = "".join([str(move + 1) for move in moves])

    c = Counter(moves)
    legal_moves = 0
    good_moves = 0

    for i in range(1, 8):
        if c[str(i)] < 6:
            score = -getBoardScore(moves + str(i))
            legal_moves += 1
            if score > 0:
                good_moves += 1
    return good_moves, legal_moves


def getBoardScore(moves):
    """

    :param moves:  moves that have been preformed given either as a string with char 1-7 or a list with values 0-6
    :return: negemax score of the board
    """

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

    board = get_board(moves)
    if Connect4Game().getGameEnded(board, -1):
        score *= -1  # we have to negate the score if the last move was a winning move

    return score


def getBoardScoreTheoretical(moves):
    """
    Returns theoretical value of the board. It is assumed the opponent plays perfect game, if we can preform a winning
    move returned score is 1 or 0 if we can hope to get a draw or -1 if opponent will win with prefect play.
    :param moves:  moves that have been preformed given either as a string with char 1-7 or a list with values 0-6
    :return:
    """
    score = getBoardScore(moves)
    if score > 0:
        score = 1
    elif score < 0:
        score = -1
    return score


def changeInTheoreticalGameValues(moves):
    # a function that returns which move (starting with 0) changed the theoretical value of first player
    # (first player did not play perfect and made a mistake which made him not win (he either drew or lost))
    gameValues = [getBoardScoreTheoretical(moves[:i + 1]) for i in range(len(moves))]

    expectedValue = -1
    for i, gameValue in enumerate(gameValues):
        if i + 1 == len(moves) and expectedValue != gameValue:
            return None
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


def errorsTillOptimalPlay(player1, nGames=10):
    """

    :param player1:  function that takes a board state as an input and returns a move (look Players in Connec4Players)
    :param nGames: number of games we play
    :return: list of lists of moves for all nGames and corresponding mistakes for each
    """
    game = Connect4Game()
    player2 = EngineConnect4Player(game).play
    players = [player2, None, player1]

    moveHistoryList = []
    mistakesList = []

    for i in range(nGames):
        board = game.getInitBoard()
        moveHistory = []
        mistakes = {}
        curPlayer = 1
        move_number = 0

        while game.getGameEnded(board, curPlayer) == 0:

            action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0

            moveHistory.append(action)

            if curPlayer == 1 and getBoardScoreTheoretical("".join([str(x + 1) for x in moveHistory])) != -1:
                mistakes[move_number] = action + 1
                action = players[0](game.getCanonicalForm(board, curPlayer))

                moveHistory[-1] = action
            board, curPlayer = game.getNextState(board, curPlayer, action)

            move_number += 1

        moveHistoryList.append(moveHistory)
        mistakesList.append(mistakes)

    return moveHistoryList, mistakesList


def saveMistakes(file, moveHistoryList, mistakesList):
    with open(file, 'w+') as f:
        for i in range(len(moveHistoryList)):
            f.write("".join([str(move + 1) for move in moveHistoryList[i]]) + '\n')
            f.write(str(mistakesList[i]) + '\n')


def openMistakes(file):
    f = open(file, 'r')
    lines = f.readlines()
    moveHistoryList = []
    mistakesList = []
    for i in range(len(lines)):
        if i % 2:
            mistakesList.append(ast.literal_eval(lines[i][:-1]))
        else:
            moveHistoryList.append(lines[i][:-1])

    return moveHistoryList, mistakesList


def countMistakesFolder(folder):
    files = []
    for r, d, f in os.walk(folder):
        for file in f:
            if 'mistakes' in file:
                files.append(file)

    avg_mistakes = {}
    for file in files:
        moveHistoryList, mistakesList = openMistakes(folder + file)
        mistakesCount = [len(mistakes) for mistakes in mistakesList]
        if 'best' not in file:
            avg_mistakes[int(file.split("_")[1].split(".")[0])] = sum(mistakesCount) / len(mistakesCount)
        else:
            avg_mistakes['best'] = sum(mistakesCount) / len(mistakesCount)

    return avg_mistakes


def weightedMistakesFolder(folder):
    weightedMistakes = {}
    for r, d, f in os.walk(folder):
        for file in f:
            if 'goodMoveAnalysis.txt' in file:
                f = open(folder + file, 'r')
                lines = f.readlines()
                if 'best' not in file:
                    weightedMistakes[int(file.split("_")[1].split(".")[0])] = float(lines[-1][:-1])
                else:
                    weightedMistakes['best'] = float(lines[-1][:-1])
    return weightedMistakes


def numberedMovesFolder(folder):
    numberedMoves = {}
    for r, d, f in os.walk(folder):
        for file in f:
            if 'numberedMoveAnalysis.txt' in file:
                f = open(folder + file, 'r')
                lines = f.readlines()
                if 'best' not in file:
                    numberedMoves[int(file.split("_")[1].split(".")[0])] = float(lines[-1][:-1])
                else:
                    numberedMoves['best'] = float(lines[-1][:-1])
    return numberedMoves


def allMatrixFolder(folder):
    c = countMistakesFolder(folder)
    w = weightedMistakesFolder(folder)
    n = numberedMovesFolder(folder)
    models = []
    ordered_count = []
    ordered_weight = []
    ordered_numbered = []

    for i in range(0, 101):
        if i in c:
            models.append(i)
            ordered_count.append(c[i])
            ordered_weight.append(w[i])
            ordered_numbered.append(n[i])

    """
    print(",".join([str(x) for x in models]))
    print(",".join([str(x) for x in ordered_count]))
    print(",".join([str(x) for x in ordered_weight]))
    print(",".join([str(x) for x in ordered_numbered]))
    """
    return models, ordered_count, ordered_weight, ordered_numbered


def bestMatrixFolder(folder):
    a = countMistakesFolder(folder)['best']
    b = weightedMistakesFolder(folder)['best']
    c = numberedMovesFolder(folder)['best']
    return a, b, c


def weightedMistakes(file):
    weightedMistakesList = []
    cumSumWeightedMistakes = []

    moveHistoryList, mistakesList = openMistakes(file)
    for i, mistakes in enumerate(mistakesList):
        mistakes_keys = mistakes.keys()
        weightedMistakes = []
        for mistake in mistakes_keys:
            good_moves, legal_moves = goodMoveProb(moveHistoryList[i][:mistake])
            weightedMistakes.append((good_moves, legal_moves))
        weightedMistakesList.append(weightedMistakes)
        cumSumWeightedMistakes.append(sum([x / y for x, y in weightedMistakes]))
    avgWeightedMistake = sum(cumSumWeightedMistakes) / len(cumSumWeightedMistakes)

    dest = file[:-9]
    with open(dest + "_goodMoveAnalysis.txt", 'w+') as f:
        f.write(str(weightedMistakesList) + '\n')
        f.write(str(cumSumWeightedMistakes) + '\n')
        f.write(str(avgWeightedMistake) + '\n')

    return weightedMistakesList, cumSumWeightedMistakes, avgWeightedMistake


def numberedMoves(file):
    moveHistoryList, mistakesList = openMistakes(file)

    nMovesList = []
    playsScore = []  # list of sums representing how good the moves were for the whole game

    for i, mistakes in enumerate(mistakesList):
        lines = []
        moves = moveHistoryList[i]
        for j in range(1, len(moves), 2):
            line = moves[:j]
            if j - 1 in mistakes:
                line = line[:-1] + str(mistakes[j - 1])
            lines.append(line)

        nMoves = []
        for line in lines:
            score, u = evaluateMove(line)
            nMoves.append((score, u))
        nMovesList.append(nMoves)
        playsScore.append(sum([x for x, _ in nMoves]))

    avg_score = sum(playsScore) / len(playsScore)

    dest = file[:-9]
    with open(dest + "_numberedMoveAnalysis.txt", 'w+') as f:
        for nMoves in nMovesList:
            f.write(str(nMoves) + '\n')
        f.write('\n')
        f.write(str(playsScore) + '\n')
        f.write('\n')
        f.write(str(avg_score) + '\n')

    return nMovesList, playsScore, avg_score


if __name__ == '__main__':
    import time
    def get_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.index' in file:
                    if 'temp' not in file:
                        files.append(file[:-6])
        return files


    g = Connect4Game()
    cpucts = [1]  # list(range(1, 7))
    for cpuct in cpucts:
        for i in [20]:

            results_folder = 'H:\\alpha-zero-trained\\final\\h2\\cpuc5_cooling\\cooling\\'+ str(i) + "\\" #"H:\\alpha-zero-trained\\final\\h1\\basic\\normal\\" + str(i) + "\\"  # "H:\\alpha-zero-trained\\final\\h0\\cpuct\\" + str(cpuct) + "\\" #"H:\\alpha-zero-trained\\final\\h2\\basic\\cooling\\"+str(i)+"\\"

            # "H:\\alpha-zero-trained\\final\\h2\\mcts_visits_1_x\\default\\1\\"

            for r, d, f in os.walk(results_folder):
                for file in f:
                    if 'mistakes' in file:
                        weightedMistakes(results_folder + file)

            """
            files = get_files(results_folder)
            print(files)

            t = time.time()
            for file in files:
                try:
                    print("Checkpoint", file)
                    # nnet players
                    n1 = NNet(g)
                    n1.load_checkpoint(results_folder, file)
                    args1 = dotdict({'numMCTSSims': 25, 'cpuct': cpuct})
                    mcts1 = MCTS(g, n1, args1)
                    player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

                    moveHistoryList, mistakesList = errorsTillOptimalPlay(player1, nGames=10)

                    saveMistakes(results_folder + file[:-8] + ".mistakes", moveHistoryList, mistakesList)
                except:
                    print("Napaka", file)
            
        #print(time.time() - t)
        """""

