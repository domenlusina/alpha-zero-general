import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from openpyxl import Workbook
from connect4.Connect4Game import Connect4Game
import numpy as np
from MCTS import MCTS
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utils import dotdict
from connect4.Connect4Logic import Board
import os
from connect4.Connect4Players import EngineConnect4Player

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.index' in file:
                if 'best' not in file and 'temp' not in file:
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


if __name__ == '__main__':
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
        """
        n1p = EngineConnect4Player(g).play
        """

        result = []
        for test_file in test_files:
            r = run_test(n1p, test_file)
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
