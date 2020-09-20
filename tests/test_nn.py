import os
import sys
import warnings

sys.path.append('/home/dlusina/alpha-zero-general/')
sys.path.append('/home/dlusina/alpha-zero-general/tests/')

warnings.filterwarnings('ignore', category=FutureWarning)
from connect4.Connect4Game import Connect4Game
import numpy as np
from MCTS import MCTS
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict
from connect4.Connect4Logic import Board

import ast


def get_files(path):
    files = []
    print(list(os.walk(path)))
    for r, d, f in os.walk(path):
        for file in f:
            if '.index' in file:
                if 'temp' not in file:
                    files.append(file[:-6])
    return files


def get_board(moves):
    moves = [int(m) - 1 for m in moves]
    b = Board()
    player = 1
    for move in moves:
        b.add_stone(move, player)
        player *= -1

    return b.np_pieces * player


def run_test(n1p, test_file_path):
    test = 0
    right_moves = 0
    with open(test_file_path, 'r') as file:
        for line in file:
            line = line.split(" ")
            moves = line[0]

            scores = [int(x) for x in line[1].split(",")]
            board = get_board(moves)

            nn_move = n1p(board)
            test += 1
            if scores[nn_move] == min(scores):
                right_moves += 1
    return right_moves / test


def preform_moves(n1p, test_file_path):
    nn_moves = []
    with open(test_file_path, 'r') as file:
        for line in file:
            line = line.split(" ")
            moves = line[0]
            board = get_board(moves)
            nn_move = n1p(board)
            nn_moves.append(nn_move)
    return nn_moves


def generateMovesTestFile(folder, model):
    test_files = ["Test_L1_R1.txt",
                  "Test_L1_R2.txt",
                  "Test_L1_R3.txt",
                  "Test_L2_R1.txt",
                  "Test_L2_R2.txt",
                  "Test_L3_R1.txt"]

    g = Connect4Game()
    nn = NNet(g)
    nn.load_checkpoint(folder, model)
    args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
    mcts1 = MCTS(g, nn, args)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    for test_file in test_files:
        moves = preform_moves(n1p, test_file)
        print(folder + model + "_" + test_file)
        with open(folder + model[:-8] + "_" + test_file, "w") as moves_file:
            moves_file.write(str(moves))


def generateMovesTestFileFolder(folder):
    files = get_files(folder) # ["best.pth.tar"]  #
    print(files)
    for file in files:
        print(file)
        generateMovesTestFile(folder, file)


def openTestResultsFile(file):
    f = open(file, 'r')
    lines = f.readlines()

    return ast.literal_eval(lines[0])


def bestMovesPercent(moves, test_file, legal_moves=None):
    test_positions = 0
    right_moves = 0
    with open(test_file, 'r') as file:
        for i, line in enumerate(file):
            if legal_moves is None or legal_moves[i] > 1:
                line = line.split(" ")

                scores = [int(x) for x in line[1].split(",")]
                test_positions += 1
                if scores[moves[i]] == min(scores):
                    right_moves += 1

    return right_moves / test_positions


def goodMovePercent(moves, test_file, legal_moves=None):
    test_positions = 0
    good_moves = 0
    with open(test_file, 'r') as file:
        for i, line in enumerate(file):
            if legal_moves is None or legal_moves[i] > 1:
                line = line.split(" ")
                scores = [int(x) for x in line[1].split(",")]
                if all([True if s == 43 or s < 0 else False for s in scores]) or all(
                        [True if s == 43 or s == 0 else False for s in scores]):
                    pass
                else:
                    if any([s < 0 for s in scores]):
                        if scores[moves[i]] < 0:
                            good_moves += 1
                        test_positions += 1
                    elif any([s == 0 for s in scores]):
                        if scores[moves[i]] == 0:
                            good_moves += 1
                        test_positions += 1

    return good_moves / test_positions


def optimalLineCumSum(moves, test_file, legal_moves=None):
    s = 0
    test_positions = 0
    with open(test_file, 'r') as file:
        for i, line in enumerate(file):
            if legal_moves is None or legal_moves[i] > 1:
                line = line.split(" ")

                scores = [int(x) for x in line[1].split(",")]

                move_score = scores[moves[i]]

                scores = list(set(scores))
                scores.sort()

                s += scores.index(move_score)
                test_positions += 1

    return s / test_positions


def getBestResults(folder):
    test_files = ["Test_L1_R1.txt",
                  "Test_L1_R2.txt",
                  "Test_L1_R3.txt",
                  "Test_L2_R1.txt",
                  "Test_L2_R2.txt",
                  "Test_L3_R1.txt"]

    legal_moves_file_lines = open('legal_moves.txt', 'r').readlines()
    legal_moves_file_lines = [ast.literal_eval(x.split('.txt ')[1]) for x in legal_moves_file_lines]

    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    best_moves = {}
    good_moves = {}
    numbered_moves = {}

    for subfolder in subfolders:
        res_best_moves = []
        res_good_moves = []
        res_numbered_moves = []
        for test_file in test_files:
            moves = openTestResultsFile(subfolder + "\\best_" + test_file)
            if test_file == "Test_L3_R1.txt":
                res_best_moves.append(bestMovesPercent(moves, test_file, legal_moves=legal_moves_file_lines[5]))
                res_good_moves.append(goodMovePercent(moves, test_file, legal_moves=legal_moves_file_lines[5]))
                res_numbered_moves.append(optimalLineCumSum(moves, test_file, legal_moves=legal_moves_file_lines[5]))
            else:
                res_best_moves.append(bestMovesPercent(moves, test_file))
                res_good_moves.append(goodMovePercent(moves, test_file))
                res_numbered_moves.append(optimalLineCumSum(moves, test_file))

        best_moves[subfolder] = res_best_moves
        good_moves[subfolder] = res_good_moves
        numbered_moves[subfolder] = res_numbered_moves

    return best_moves, good_moves, numbered_moves


def getAllResults(folder):
    test_files = ["Test_L1_R1.txt",
                  "Test_L1_R2.txt",
                  "Test_L1_R3.txt",
                  "Test_L2_R1.txt",
                  "Test_L2_R2.txt",
                  "Test_L3_R1.txt"]

    legal_moves_file_lines = open('legal_moves.txt', 'r').readlines()
    legal_moves_file_lines = [ast.literal_eval(x.split('.txt ')[1]) for x in legal_moves_file_lines]

    res_best_moves = {}
    res_good_moves = {}
    res_optimal_line = {}

    for i, test_file in enumerate(test_files):
        res_best_moves[test_file] = {}
        res_good_moves[test_file] = {}
        res_optimal_line[test_file] = {}
        for r, d, f in os.walk(folder):
            for file in f:
                if test_file in file and 'best' not in file:
                    moves = openTestResultsFile(folder + file)
                    model = int(file.split('_')[1])

                    res_best_moves[test_file][model] = bestMovesPercent(moves, test_file,
                                                                        legal_moves=legal_moves_file_lines[i])
                    res_good_moves[test_file][model] = goodMovePercent(moves, test_file,
                                                                       legal_moves=legal_moves_file_lines[i])
                    res_optimal_line[test_file][model] = optimalLineCumSum(moves, test_file,
                                                                           legal_moves=legal_moves_file_lines[i])

    return res_best_moves, res_good_moves, res_optimal_line


if __name__ == '__main__':
    import time

    t = time.time()
    for i in [1]:
        folder = "H:\\alpha-zero-trained\\final\\h2\\basic\\cooling\\20\\"
        # folder = "C:\\Magistrsko_delo\\alpha-zero-general\\h2\\cpuct5\\default\\"+str(i)+"\\"
        # "C:\\Magistrsko_delo\\alpha-zero-general\\tanh\\"+str(i)+"\\"  # "H:\\alpha-zero-trained\\final\\h0\\cpuct\\" + str(i) + "\\"
        generateMovesTestFileFolder(folder)
        print(time.time() - t)

    """
    g = Connect4Game()
    title = "test"

    nn = NNet(g)
    results = []
    folder = 'H:\\alpha-zero-trained\\h0\\newresblock\\'  # 'H:\\alpha-zero-trained\\h0\\window\\'
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    test_files = ["Test_L1_R1.txt",
                  "Test_L1_R2.txt",
                  "Test_L1_R3.txt",
                  "Test_L2_R1.txt",
                  "Test_L2_R2.txt",
                  "Test_L3_R1.txt"]

    for subfolder in subfolders:
        nn.load_checkpoint(subfolder, 'best.pth.tar')
        args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
        mcts1 = MCTS(g, nn, args)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

        result = []
        for test_file in test_files:
            r = preform_moves(n1p, test_file)
            result.append(r)
        results.append([subfolder.split("\\")[-1]] + result)

    book = Workbook()
    sheet = book.active
    sheet.append([' '] + [test_file[5:-4] for test_file in test_files] + ["Average"])
    print(results)
    for row in results:
        if len(row) > 0:
            row.append(sum(row[1:]) / (len(row) - 1))
        sheet.append(row)

    book.save(folder + title + ".xlsx")
    """
"""
if __name__ == '__main__':
    g = Connect4Game()
    title = "h2_combined_cooling"

    nn = NNet(g)
    results = []
    folder = 'H:\\alpha-zero-trained\\h2_combined_cooling\\'  # 'H:\\alpha-zero-trained\\h0\\window\\'
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    test_files = ["Test_L1_R1.txt",
                  "Test_L1_R2.txt",
                  "Test_L1_R3.txt",
                  "Test_L2_R1.txt",
                  "Test_L2_R2.txt",
                  "Test_L3_R1.txt"]

    for subfolder in subfolders:
        files = get_files(subfolder)
        files = sorted(files, key=lambda file: int(file[file.find('_') + 1: file.find('.')]))
        ind = [int(file[file.find('_') + 1: file.find('.')]) for file in files]

        for file in files:
            nn.load_checkpoint(subfolder, file)
            args = dotdict({'numMCTSSims': 25, 'cpuct': 5})
            mcts1 = MCTS(g, nn, args)
            n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
            
            result = []
            for test_file in test_files:
                r = run_test(n1p, test_file)
                result.append(r)
            results.append([file] + result)

    book = Workbook()
    sheet = book.active
    sheet.append([' '] + [test_file[5:-4] for test_file in test_files] + ["Average"])
    print(results)
    for row in results:
        if len(row) > 0:
            row.append(sum(row[1:]) / (len(row) - 1))
        sheet.append(row)

    book.save(title + ".xlsx")
"""
