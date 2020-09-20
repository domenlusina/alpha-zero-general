import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from connect4.Connect4Logic import Board
from tests.test_nn import get_board

plt.style.use('seaborn-colorblind')


def drawOddEvenThreats(data, title, ymax, save_as=None):
    keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    plt.title(title)
    plt.ylabel("Povprečno število groženj")
    plt.xlabel("Iteracije učenja")

    x = [int(key.split("_")[1]) for key in keys]
    plt.xlim(0, 100)
    plt.ylim(0, ymax)

    labels = ['1. lihe', '1. sode', '2. lihe', '2. sode']
    for i in range(4):
        plt.plot(x, [data[key][i] for key in keys], label=labels[i])

    plt.legend(title="Grožnje")

    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def drawLine(data, title, ytitle, ymax, save_as=None):
    keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel("Iteracije učenja")

    plt.xlim(0, 100)
    plt.ylim(0, ymax)

    x = [int(key.split("_")[1]) for key in keys]
    plt.plot(x, [data[key] for key in keys])

    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


@njit
def major_threats_line(board, transpose=False):
    if transpose:
        board = board.transpose()

    threats = []

    for i in range(board.shape[0]):
        for j in range(board.shape[1] - 3):
            tmp = board[i, j:j + 4]
            if tmp.sum() == 3:
                threats.append((i, j + np.argmin(tmp)))

    if transpose:
        threats = [(y, x) for x, y in threats]

    return threats


def major_threats_diag(board, transpose=False):
    if transpose:
        board = np.fliplr(board)
    threats = major_threats_diag_(board)
    if transpose:
        threats = [(6 - x, y) for x, y in threats]
    return threats


def major_threats_diag_(board):
    threats = []
    for i in range(board.shape[0] - 3):
        for j in range(board.shape[1] - 3):
            tmp = np.array([board[i, j], board[i + 1, j + 1], board[i + 2, j + 2], board[i + 3, j + 3]])
            if tmp.sum() == 3:
                offset = np.argmin(tmp)
                threats.append((i + offset, j + offset))

    return threats


def getAllThreatsCurrentPlayer(board):
    curPlayerThreats = []

    curPlayerThreats.extend(major_threats_line(board))
    curPlayerThreats.extend(major_threats_line(board, transpose=True))

    curPlayerThreats.extend(major_threats_diag(board))
    curPlayerThreats.extend(major_threats_diag(board, transpose=True))

    curPlayerThreats = list(set(curPlayerThreats))
    return curPlayerThreats


def filterSameParityThreats(threats):
    lowestOddThreat = np.ones(7) * (-np.inf)
    lowestEvenThreat = np.ones(7) * (-np.inf)
    for threat in threats:
        trow, tcol = threat
        if trow % 2 == 0:
            if lowestOddThreat[tcol] < trow:
                lowestOddThreat[tcol] = trow
        else:
            if lowestEvenThreat[tcol] < trow:
                lowestEvenThreat[tcol] = trow

    res = []
    for i in range(7):
        if lowestOddThreat[i] != -np.inf:
            res.append((lowestOddThreat[i], i))
        if lowestEvenThreat[i] != -np.inf:
            res.append((lowestEvenThreat[i], i))
    return res


def getAllThreats(moves):
    cannonicalBoard = get_board(moves)
    curPlayerThreats = filterSameParityThreats(getAllThreatsCurrentPlayer(cannonicalBoard))

    enemyPlayerThreats = filterSameParityThreats(getAllThreatsCurrentPlayer(-1 * cannonicalBoard))
    return curPlayerThreats, enemyPlayerThreats


def optimalFirstMove(moves):
    return moves[0] == '4'


def blockedEnemyWin(moves):
    board = Board()
    curPlayer = 1
    blocked = []
    winnable_moves = []
    for i, move in enumerate(moves):
        tmp = winnableMove(board)
        winnable_moves.extend(tmp)
        m = int(move) - 1
        board.add_stone(m, -curPlayer)
        # check if enemy player could have won by playing the move we did
        if board.get_win_state().winner == -curPlayer:
            if len(tmp) == 0:
                print(board.np_pieces)
            blocked.append(i)
        last_move, = np.where(board.np_pieces[:, m] != 0)
        board.np_pieces[last_move[0], m] = curPlayer

        curPlayer *= -1

    return blocked, winnable_moves


def winnableMove(board):
    wm = []

    if board.np_pieces.sum() == 0:
        curPlayer = -1
    else:
        curPlayer = 1

    for m, legal in enumerate(board.get_legal_moves()):
        if legal:
            board.add_stone(m, curPlayer)

            last_move, = np.where(board.np_pieces[:, m] != 0)

            if board.get_win_state().winner is not None:
                wm.append((last_move[0], m))

            board.np_pieces[last_move[0], m] = 0
    return wm


def missedWins(moves):
    missed = []
    board = Board()
    detectedMissedWins = [[], [], []]
    curPlayer = 1

    for i, move in enumerate(moves[:-2]):
        m = int(move) - 1

        w_moves = winnableMove(board)
        if len(w_moves) >= 1:
            for win_move in w_moves:
                if win_move[1] != m and win_move not in detectedMissedWins[1 + curPlayer]:
                    detectedMissedWins[1 + curPlayer].append(win_move)
                    missed.append((i, win_move))

        board.add_stone(m, curPlayer)

        curPlayer *= -1
    return missed


def readMoves(file):
    with open(file) as f:
        content = f.readlines()

    return [line[:-1] for line in content]


def getSelfPlayFiles(folder):
    res = []
    for r, d, f in os.walk(folder):
        for file in f:
            if 'self_play' in file:
                res.append(file)
    return res


if __name__ == "__main__":
    folders = ["C:\\Magistrsko_delo\\alpha-zero-general\\h0\\cpuct\\1\\",
               "C:\\Magistrsko_delo\\alpha-zero-general\\h0\\cpuct\\5\\",
               "C:\\Magistrsko_delo\\alpha-zero-general\\h2\\cpuct5\\default\\1\\",
               "H:\\alpha-zero-trained\\final\\h2\\mcts_visits_tanh\\default\\1\\"]

    titles = ["Brez hevristike, cpuct=1",
              "Brez hevristike, cpuct=5",
              "S hevristike, cpuct=5",
              "S hevristike, cpuct=1"
              ]
    t = ["cpuct=1",
         "cpuct=5",
         "heur_cpuct=5",
         "heur_cpuct=1"]
    for j in range(len(folders)):

        title = titles[j]
        folder = folders[j]

        save_folder = "C:\\Users\\Domen\\Desktop\\tmp\\"

        files = getSelfPlayFiles(folder)

        all_files_threats = {}
        sorted_files = sorted(files, key=lambda x: int(x.split("_")[1]))
        """
        for file in files:
            moves_games = readMoves(folder + file)
            results = []
            for moves in moves_games:
                first_player = []
                second_player = []
                for i in range(2, len(moves), 2):
                    board = -1 * get_board(moves[:(i - 1)])
                    fp = getAllThreatsCurrentPlayer(board)
                    board = -1 * get_board(moves[:i])
                    sp = getAllThreatsCurrentPlayer(board)
                    first_player.extend(fp)
                    second_player.extend(sp)
                first_player = list(set(first_player))
                second_player = list(set(second_player))
                results.append((first_player, second_player))
            all_files_threats[file] = results

        all_counted_threats = {}
        for key in all_files_threats:
            fp_odd = 0
            fp_even = 0
            sp_odd = 0
            sp_even = 0
            for threats in all_files_threats[key]:
                for el in threats[0]:
                    if el[0] % 2 == 0:
                        fp_even += 1
                    else:
                        fp_odd += 1
                for el in threats[1]:
                    if el[0] % 2 == 0:
                        sp_even += 1
                    else:
                        sp_odd += 1

            all_counted_threats[key] = [fp_odd, fp_even, sp_odd, sp_even]
        """
        all_blocked = {}
        for file in files:
            moves_games = readMoves(folder + file)
            results = [[], []]
            for moves in moves_games:
                tmp = blockedEnemyWin(moves)
                results[0].extend(tmp[0])
                results[1].extend(tmp[1])
            if len(results[1]) == 0:
                print(len(results[0]), len(results[1]))
                all_blocked[file] = 1
            else:
                all_blocked[file] = len(results[0]) / len(results[1])

        all_missed_wins = {}
        for file in files:
            moves_games = readMoves(folder + file)
            results = []
            for moves in moves_games:
                results.extend(missedWins(moves))
            all_missed_wins[file] = len(results)

        # drawOddEvenThreats(all_counted_threats, title, 400, save_as=save_folder + t[j] + "threats.png")
        drawLine(all_blocked, title, "Relativno število blokiranih zmag", 1.05, save_as=save_folder + t[j] + "blocked_rel.png")
        drawLine(all_missed_wins, title, "Število spregledanih zmag", 150, save_as=save_folder + t[j] + "missed.png")

"""
def minor_threats_line(board, transpose=False):
    if transpose:
        board = board.transpose()

    minor = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],

    ])

    threats = []

    for i in range(board.shape[0]):
        for j in range(board.shape[1] - 3):
            tmp = board[i, j:j + 4]
            if (minor == tmp).all(axis=1).any():
                minor_threats = np.where(tmp == 0)[0]
                threats.append((i, j + minor_threats[0]))
                threats.append((i, j + minor_threats[1]))

    if transpose:
        threats = [(y, x) for x, y in threats]

    return threats


def minor_threats_diag(board, transpose=False):
    if transpose:
        board = board.transpose()

    minor = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],

    ])

    threats = []

    for i in range(board.shape[0] - 3):
        for j in range(board.shape[1] - 3):
            tmp = board[(i, i + 1, i + 2, i + 3), (j, j + 1, j + 2, j + 3)]
            if (minor == tmp).all(axis=1).any():
                minor_threats = np.where(tmp == 0)[0]
                threats.append((i + minor_threats[0], j + minor_threats[0]))
                threats.append((i + minor_threats[1], j + minor_threats[1]))

    if transpose:
        threats = [(y, x) for x, y in threats]

    return threats
    
def filter_minor_threats(minor_threats, major_threats):
    res = []
    for i in range(0, len(minor_threats), 2):
        if (not minor_threats[i] in major_threats) and (not minor_threats[i + 1] in major_threats):
            res.append(minor_threats[i])
            res.append(minor_threats[i + 1])
    return res
    
"""
