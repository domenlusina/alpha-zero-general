import numpy as np
from numba import njit, b1, i1, int64, float64

"""
@jitclass([
    ('board', types.int8[:, :]),
    ('row', types.int64),
    ('col', types.int64),
    ('value', types.int64)
])
"""


class Node:
    def __init__(self, board, row, col):
        # self.children = np.zeros((7, 2), dtype=np.int)
        self.board = board
        self.row = row
        self.col = col


@njit(b1(i1[:, :], i1, i1))
def was_winning_move(board, row, col):
    if col == -1:
        return False

    player = board[row, col]
    player_pieces = board == player
    win_len = 4
    row_win = player_pieces[row, :]
    for i in range(row_win.size - win_len + 1):
        if row_win[i: i + win_len].all():
            return True

    diag_win1 = np.diag(player_pieces, col - row)
    for i in range(diag_win1.size - win_len + 1):
        if diag_win1[i: i + win_len].all():
            return True
    new_col = 6 - col
    diag_win2 = np.diag(player_pieces[:, ::-1], new_col - row)
    for i in range(diag_win2.size - win_len + 1):
        if diag_win2[i: i + win_len].all():
            return True

    if row < 3:
        col_win = player_pieces[row:, col]
        for i in range(col_win.size - win_len + 1):
            if col_win[i: i + win_len].all():
                return True

    return False


@njit(i1[:, :](i1[:, :], i1, i1, i1))
def add_stone(board, row, col, player):
    # available_idx = np.argmin(board[:, column] == 0) - 1
    new_board = board.copy()
    new_board[row][col] = player

    return new_board


@njit(i1(i1[:, :]))
def player2move(board):
    player = -1
    if np.count_nonzero(board) % 2 == 0:
        player = 1

    return player


@njit(b1[:](i1[:, :]))
def valid_moves(board):
    return board[0] == 0


@njit(i1(i1[:, :], i1))
def playable_row(board, col):
    return np.where(board[:, col] == 0)[0][-1]


@njit(float64(i1[:, :], int64, int64, int64, float64, float64, int64, float64[:, :]))
def minimax(board, move_row, move_col, depth, alpha, beta, player, result):
    if was_winning_move(board, move_row, move_col):
        return -player

    moves = np.where(valid_moves(board))[0]  # better order moves from center shuffle?
    if depth == 0 or moves.size == 0:
        return 0

    if player == 1:
        best_val = -np.inf
    else:
        best_val = np.inf

    for i in range(moves.size):
        col = moves[i]
        row = playable_row(board, col)  # np.argmin(board[:, col] == 0) - 1
        new_board = add_stone(board, row, col, player)
        # child_node = Node(new_board, row, col)

        value = minimax(new_board, row, col, depth - 1, alpha, beta, -player, result)
        if move_row == -1 and move_col == -1:
            result[col, 0] = 1
            result[col, 1] = value
            # root_children.append((col, value))

        if player == 1:
            best_val = max(best_val, value)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, value)
            alpha = min(alpha, best_val)

        if beta <= alpha:
            break
    return best_val


# @njit(int64(i1[:, :], int64))
def best_move_alpha_beta(board, depth):
    board = board.astype(np.int8)
    # root = Node(board, -1, -1)
    result = np.zeros((7, 2))
    v = minimax(board, -1, -1, depth, -np.inf, np.inf, 1, result)  # board is in canonical form
    moves = []

    for move in range(7):
        searched, value = result[move, :]
        if searched and value == v:
            moves.append(move)

    return np.random.choice(moves)


if __name__ == "__main__":
    import time

    b = np.zeros((6, 7))
    b[5][3] = 1
    b[5][2] = 1
    # b[5][1] = 1
    b[4][3] = -1

    b = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]])
    b = b.astype(np.int8)
    b = b * -1

    # b[4][2] = -1
    # b[4][1] = -1
    # print(was_winning_move(b, 2,1))

    print(b)
    t = time.time()
    print(best_move_alpha_beta(b, 9))
    print(time.time() - t)
