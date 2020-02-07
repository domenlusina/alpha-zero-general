import os
from subprocess import run, PIPE

import numpy as np

from connect4.Connect4Heuristics import heuristic1player, heuristic2
from connect4.Connect4Tree import best_move_alpha_beta
# IMPORTANT to know: when using getCanonicalForm, tokens of the current player are marked with 1
# board given to a player is always in canonical form
# during self play the first player's tokens are 1's and second player tokens are -1

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanConnect4Player:
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


class EngineConnect4Player:
    def __init__(self, game):
        self.game = game
        self.path = "..\\connect4\\bin\\best_move.exe"

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

        return b_move - 1


class OneStepLookaheadConnect4Player:
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


class AlphaBetaConnect4Player:
    def __init__(self, game, depth):
        self.game = game
        self.depth = depth

    def play(self, board):
        return best_move_alpha_beta(board, self.depth)


class HeuristicOneConnect4Player:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        return heuristic1player(board, 1)


class HeuristicTwoConnect4Player:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        return heuristic2(board)


class VelenaConnect4Player:
    def __init__(self, game):
        self.game = game
        self.path = "..\\veleng\\Debug\\"

    def get_exposed_tokens(self, board, player):
        non_empty_fields = board != 0
        row_idx = np.argmax(non_empty_fields, axis=0)
        results = []
        for i, j in enumerate(row_idx):
            if board[j, i] == player:
                results.append((j, i))
        return results

    def move_seq(self, board, player):
        if not board.any():
            return []

        exposed_tokens = self.get_exposed_tokens(board, player)
        if not exposed_tokens:
            return None

        for exposed_token in exposed_tokens:
            board[exposed_token] = 0
            res = self.move_seq(board, -player)
            if res is not None:
                seq = [exposed_token[1]]
                seq.extend(res)
                return seq

            board[exposed_token] = player

    def play(self, board):
        print(board)
        if not board.any():
            moves = '0'
        else:
            m = self.move_seq(board.copy(), -1)
            m.reverse()
            m.append(-1)
            moves = "".join(map(lambda x: str(x + 1), m))

        cwd = os.getcwd()
        os.chdir(self.path)

        process = run(os.getcwd() + '\\veleng.exe', stdout=PIPE, stderr=PIPE, input=(moves + '\nq\n').encode())
        out = process.stdout.decode()
        res = out.split('\n')[1].split(' ')
        if '\r' in res:
            res.remove('\r')

        best_moves = list(map(lambda x: int(x) - 1, res))

        # just double check that we output valid moves only
        illegal_moves = board[0] != 0
        for move, is_illegal in enumerate(illegal_moves):
            if is_illegal and move in best_moves:
                best_moves.remove(move)

        os.chdir(cwd)

        print(best_moves)
        return np.random.choice(best_moves)
