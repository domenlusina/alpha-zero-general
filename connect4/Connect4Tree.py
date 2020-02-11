import numpy as np
from numba import njit, b1, i1
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
cimport cython

class Node:
    def __init__(self, board, move):
        self.children = []
        self.board = board
        self.move = move
        self.value = 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
def was_winning_move(np.ndarray[np.int_t, ndim=2] board, int row, int col):
    if col == -1:
        return False

    cdef int player = board[row, col]
    player_pieces = np.where(board==player, True, False)
    cdef int win_len = 4

    row_win = player_pieces[row, :]
    cdef int i
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


def add_stone( board, int row, int col, int player):
    # available_idx = np.argmin(board[:, column] == 0) - 1
    cdef  new_board = board.copy()
    new_board[row][col] = player

    return new_board

"""
def player2move(board):
    player = -1
    if np.count_nonzero(board) % 2 == 0:
        player = 1

    return player
"""

def valid_moves( board):
    return board[0] == 0


def playable_row( board, int col):
    return np.where(board[:, col] == 0)[0][-1]


def minimax(node, int depth, alpha, beta, int player):
    was_win = was_winning_move(node.board, node.move[0], node.move[1])
    if was_win:
        node.value = -player
        return node.value

    moves = np.where(valid_moves(node.board))[0]  # better order moves from center shuffle?
    if depth == 0 or moves.size == 0:
        return node.value

    if player == 1:
        best_val = -np.inf
    else:
        best_val = np.inf
    for col in moves:
        row = playable_row(node.board, col)  # np.argmin(node.board[:, col] == 0) - 1
        new_board = add_stone(node.board, row, col, player)
        child_node = Node(new_board, (row, col))
        node.children.append(child_node)

        value = minimax(child_node, depth - 1, alpha, beta, -player)

        if player == 1:
            best_val = max(best_val, value)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, value)
            alpha = min(alpha, best_val)

        node.value = best_val
        if beta <= alpha:
            break
    return best_val


def best_move_alpha_beta( board, depth):
    board = board.astype(np.int)
    root = Node(board, (-1, -1))
    v = minimax(root, depth, -np.inf, np.inf, 1)  # board is in canonical form
    moves = []

    for child in root.children:
        if child.value == v:
            moves.append(child.move[1])

    return np.random.choice(moves)

