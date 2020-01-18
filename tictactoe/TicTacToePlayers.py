import numpy as np

from tictactoe.TicTacToeLogic import Board
from tictactoe.transform import Transform, Identity, Rotate90, Flip

TRANSFORMATIONS = [Identity(), Rotate90(1), Rotate90(2), Rotate90(3),
                   Flip(np.flipud), Flip(np.fliplr),
                   Transform(Rotate90(1), Flip(np.flipud)),
                   Transform(Rotate90(1), Flip(np.fliplr))]

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class OptimalPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # valid = self.game.getValidMoves(board, 1)
        b = Board(pieces=board)
        b_arr = np.array(board)
        player = int(np.sum(b_arr)) % 2 * (-2) + 1

        # if we can win we win
        w = can_win(b, player)
        if w:
            return w

        # we have to prevent a defeat
        l = can_win(b, -player)
        if l:
            return l

        if np.all(b_arr == np.zeros((3, 3))):
            if np.random.choice([True, False]):
                return 4
            else:
                return np.random.choice([0, 2, 6, 8])

        if board_match(b_arr, np.array([[1, 0, 0], [0, -1, 1], [0, 0, 0]])):
            if (b_arr[0][2] == 1 and b_arr[1][0] == 1) or (b_arr[0][1] == 1 and b_arr[2][0] == 1):
                return 0
            elif (b_arr[0][0] == 1 and b_arr[1][2] == 1) or (b_arr[0][1] == 1 and b_arr[2][2] == 1):
                return 2
            elif (b_arr[0][0] == 1 and b_arr[2][1] == 1) or (b_arr[1][0] == 1 and b_arr[2][2] == 1):
                return 6
            elif (b_arr[0][2] == 1 and b_arr[2][1] == 1) or (b_arr[1][2] == 1 and b_arr[2][0] == 1):
                return 8

        # center is played so we play corner
        if board_match(b_arr, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])):
            return 0
        if board_match(b_arr, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])):
            return 4

        # we can fork if edge and center are played
        if board_match(b_arr, np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])):
            return 0

        # if center and corner are played we play other corner
        if board_match(b_arr, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])):
            if board[0][0] == -1:
                return 8
            elif board[0][2] == -1:
                return 6
            elif board[2][0] == -1:
                return 2
            elif board[2][2] == -1:
                return 0

        # if edge, center and corner are played we can win by playing a corner furthest away from edge man
        if board_match(b_arr, np.array([[-1, 0, 0], [0, 1, -1], [0, 0, 1]])):
            if b_arr[0][0] == -1 and b_arr[1][2] == -1:
                return 6
            elif b_arr[0][0] == -1 and b_arr[2][1] == -1:
                return 2

            elif b_arr[0][2] == -1 and b_arr[1][0] == -1:
                return 8
            elif b_arr[0][2] == -1 and b_arr[2][1] == -1:
                return 0

            elif b_arr[2][0] == -1 and b_arr[0][1] == -1:
                return 8
            elif b_arr[2][0] == -1 and b_arr[1][2] == -1:
                return 0

            elif b_arr[0][1] == -1 and b_arr[2][2] == -1:
                return 6
            elif b_arr[1][0] == -1 and b_arr[2][2] == -1:
                return 2

        # edge next to the corner is played
        if board_match(b_arr, np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0]])):
            return 4

        # both players played the corner on same sides
        if board_match(b_arr, np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]])):
            if b_arr[0][0] == 1:
                return 8
            elif b_arr[0][2] == 1:
                return 6
            elif b_arr[2][0] == 1:
                return 2
            elif b_arr[2][2] == 1:
                return 0

        # edge away from the corner is played
        if board_match(b_arr, np.array([[1, 0, 0], [0, 0, 0], [0, -1, 0]])):
            return 4

        # both players played the corner on opposite sides -> we play a corner
        if board_match(b_arr, np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])) or \
                board_match(b_arr, np.array([[1, -1, 1], [0, 0, 0], [0, 0, -1]])):
            if b_arr[0][0] == 0:
                return 0
            elif b_arr[0][2] == 0:
                return 2
            elif b_arr[2][0] == 0:
                return 6
            elif b_arr[2][2] == 0:
                return 8

        # we have to create a thread by playing edge so we avoid a fork
        if board_match(b_arr, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])):
            return np.random.choice([1, 3, 5, 7])

        if b_arr[1][1] == 0:
            return 4
        elif b_arr[0][0] == 0:
            return 0
        elif b_arr[0][2] == 0:
            return 2
        elif b_arr[2][0] == 0:
            return 6
        elif b_arr[2][2] == 0:
            return 8
        elif b_arr[0][1] == 0:
            return 1
        elif b_arr[1][0] == 0:
            return 3
        elif b_arr[1][2] == 0:
            return 5
        elif b_arr[2][1] == 0:
            return 7


def board_match(b1, b2):
    for t in TRANSFORMATIONS:
        if np.all(t.transform(b1) == b2):
            return True
    return False


def can_win(board, player):
    moves = board.get_legal_moves(player)
    for move in moves:
        b = Board(pieces=board.pieces)
        b.execute_move(move, player)
        if b.is_win(player):
            return move[0] * b.n + move[1]


class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x
            # a = raw_input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
