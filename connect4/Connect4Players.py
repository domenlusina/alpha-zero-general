from subprocess import run, PIPE

import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanConnect4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        print('\nMoves:', [i for (i, valid) in enumerate(valid_moves) if valid])

        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                print('Invalid move')
        return move


class EngineConnect4Player():
    def __init__(self, game):
        self.game = game
        self.path = "C:\\Magistrsko_delo\\connect4\\bin\\best_move.exe"

    def position_param(self, board):
        no_moves = np.count_nonzero(board)

        mask = ""
        for i in range(board.shape[1]):
            for x in range(board.shape[0]):
                j = board.shape[0] - x - 1
                if board[j, i] != 0:
                    mask += "1"
                else:
                    mask += "0"
            mask += "0"

        mask = ''.join(reversed(mask))

        current_position = ""
        for i in range(board.shape[1]):
            for x in range(board.shape[0]):
                j = board.shape[0] - x - 1
                if board[j, i] == 1:
                    current_position += "1"
                else:
                    current_position += "0"
            current_position += "0"

        current_position = ''.join(reversed(current_position))

        return int(current_position, 2), int(mask, 2), no_moves

    def play(self, board):
        param = self.position_param(board)
        param = " ".join(map(str, param))

        process = run(self.path, stdout=PIPE, input=(param + "\n").encode())
        b_move = process.returncode

        return b_move-1


class OneStepLookaheadConnect4Player():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""

    def __init__(self, game, verbose=False):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % self.game.stringRepresentation(board))

        return ret_move
