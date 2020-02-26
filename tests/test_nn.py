import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from openpyxl import Workbook
from connect4.Connect4Game import Connect4Game
import numpy as np
from MCTS import MCTS
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utils import dotdict
from connect4.Connect4Logic import Board
from connect4.Connect4Players import RandomPlayer


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
    title = "H0_cpuct"

    nn = NNet(g)
    results = []

    for i in range(1, 7):
        folder = 'H:\\alpha-zero-trained\\h0\\cpuct\\'+str(i)
        nn.load_checkpoint(folder, 'best.pth.tar')
        args = dotdict({'numMCTSSims': 25, 'cpuct': i})
        mcts1 = MCTS(g, nn, args)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

        test_files = ["Test_L1_R1.txt",
                      "Test_L1_R2.txt",
                      "Test_L1_R3.txt",
                      "Test_L2_R1.txt",
                      "Test_L2_R2.txt",
                      "Test_L3_R1.txt"]

        result = []
        for test_file in test_files:
            r = run_test(n1p, test_file)
            result.append(r)
        results.append([folder.split('\\')[-1]]+result)

    book = Workbook()
    sheet = book.active
    sheet.append([' '] + [test_file[5:-4] for test_file in test_files] + ["Average"])
    print(results)
    for row in results:
        if len(row)>0:
            row.append(sum(row) / len(row))
        sheet.append(row)

    book.save(title + ".xlsx")


