import numpy as np
from numba import njit


def heuristic1lookahead(board):
    board = np.copy(board)
    player = 1
    # if np.sum(board) == 0:
    #    player = 1

    valid_moves = board[0] == 0
    l_len = 4
    for j in range(valid_moves.size):
        if valid_moves[j]:

            available_idx, = np.where(board[:, j] == 0)
            board[available_idx[-1]][j] = player

            # we look if we can win in one move
            player_pieces = board * player

            f1l = straight_lines(player_pieces, l_len, 4)
            f1c = straight_lines(player_pieces, l_len, 4, transpose=True)
            f1d = diagonal_lines(player_pieces, l_len, 4)
            board[available_idx[-1]][j] = 0
            if f1l or f1c or f1d:
                return j

    return None


def heuristic1(board):
    player = 1
    # if np.sum(board) == 0:
    #     player = 1
    res = np.zeros(7)
    res[heuristic1player(board, player)] = 1
    return res


def heuristic1player(board, player):
    valid_moves = board[0] == 0
    scores = np.zeros(valid_moves.size)
    valid_moves_scores = []

    # feature 1
    win_move = -1
    prevent_win_move = -1
    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(board[:, j] == 0)
            board[available_idx[-1]][j] = player
            player_score = board_score(board, player, [1])
            if player_score == np.inf:
                win_move = j

            board[available_idx[-1]][j] = -player
            enemy_score = board_score(board, -player, [1])
            if enemy_score == np.inf:
                prevent_win_move = j
            board[available_idx[-1]][j] = 0

    if win_move != -1:
        return win_move

    if prevent_win_move != -1:
        return prevent_win_move

    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(board[:, j] == 0)
            board[available_idx[-1]][j] = player

            scores[j] = board_score(board, player, [2, 3, 4])
            if scores[j] != np.inf:
                board[available_idx[-1]][j] = -player
                scores[j] -= board_score(board, -player, [2, 3, 4])

            board[available_idx[-1]][j] = 0  # we undo the move
            valid_moves_scores.append(scores[j])
        else:
            scores[j] = -np.inf

    if np.max(scores) == -np.inf:
        return np.argmax(valid_moves)

    return np.argmax(scores)


def board_score(board, player, features):
    board = np.copy(board)
    score = 0
    player_pieces = board * player
    shape = board.shape
    l_len = 4

    """FEATURE 1"""
    if 1 in features:
        f1l = straight_lines(player_pieces, l_len, 4)
        f1c = straight_lines(player_pieces, l_len, 4, transpose=True)
        f1d = diagonal_lines(player_pieces, l_len, 4)

        if f1l or f1c or f1d:
            return np.inf

    """FEATURE 2"""
    if 2 in features:
        # inf cases
        f2l = straight_lines(player_pieces, l_len, 3)
        for i, j in f2l:
            if j < shape[1] - 4:
                if playable(board, (i, j)) and playable(board, (i, j + 4)):  # left field and right field are playable
                    return np.inf

        f2d = diagonal_lines(player_pieces, l_len, 3)
        for (i, j), up in f2d:
            if playable(board, (i, j)):  # left field is playable
                if j + 4 < shape[1] and shape[0] > i + up * 4 >= 0:
                    if playable(board, (i + up * 4, j + 4)):
                        return np.inf

        # preventable win cases - line
        for i, j in f2l:
            for y in range(j, j + 4):
                if playable(board, (i, y)):
                    score += 900000
                    break
                    # perhaps there should be some reward even if it is not immediately playable

        # preventable win cases - column

        f2c = [(i, j) for i, j in straight_lines(player_pieces, l_len, 3, transpose=True) if i < shape[0] - 3]

        score += 900000 * len(f2c)

        # preventable win cases diagonal
        for (i, j), up in f2d:
            for y in range(4):
                if playable(board, (i + up * y, j + y)):
                    score += 900000
                    break

    """FEATURE 3"""

    if 3 in features:
        # case 1 - line
        f3l = straight_lines(player_pieces, l_len, 2)
        for i, j in f3l:
            if playable(board, (i, j)):  # left field is empty
                if playable(board, (i, j + 3)):  # right field is empty
                    score += 50000

        # case 1 - diag
        f3d = diagonal_lines(player_pieces, l_len, 2)
        for (i, j), up in f3d:
            if playable(board, (i, j)):  # left field is empty
                if playable(board, (i + up * 3, j + 3)):  # right field is empty
                    score += 50000

        # case 2 - line
        for i, j in f3l:
            if playable(board, (i, j)) and playable(board, (i, j + 1)):
                score += 10000
                for y in range(j - 1, -1, -1):
                    if playable(board, (i, y)):
                        score += 10000
                    else:
                        break
            elif playable(board, (i, j + 2)) and playable(board, (i, j + 3)):
                score += 10000
                for y in range(j + 4, shape[1]):
                    if playable(board, (i, y)):
                        score += 10000
                    else:
                        break

        # case 2 - diag
        for (i, j), up in f3d:
            if playable(board, (i, j)) and playable(board, (i + up, j + 1)):
                score += 10000
                x = i - up
                for y in range(j - 1, -1, -1):
                    if 0 <= x < shape[0] and playable(board, (x, y)):
                        score += 10000
                        x -= up
                    else:
                        break
            elif playable(board, (i + 2 * up, j + 2)) and playable(board, (i + 3 * up, j + 3)):
                score += 10000
                x = i + 4 * up
                for y in range(j + 4, shape[1]):
                    if 0 <= x < shape[0] and playable(board, (x, y)):
                        score += 10000
                        x += up
                    else:
                        break

        # case 2 - column
        f3c = [(i, j) for i, j in straight_lines(player_pieces, l_len, 2, transpose=True) if i < shape[0] - 3]
        score += len(f3c) * 10000

    if 4 in features:
        """FEATURE 4 """
        for i in range(shape[0]):
            for j in range(shape[1]):
                if player_pieces[i][j] == 1 and is_isolated(player_pieces, (i, j)):
                    if j == 0 or j == 6:
                        score += 40
                    elif j == 1 or j == 5:
                        score += 70
                    elif j == 2 or j == 4:
                        score += 120
                    elif j == 3:
                        score += 200
    return score


def playable(board, field):
    if board[field[0]][field[1]] != 0:
        return False
    if field[0] == board.shape[0] - 1:
        return True
    if board[field[0] + 1][field[1]] != 0:
        return True
    return False


def straight_lines(player_pieces, l_len, no_pieces, transpose=False):
    if transpose:
        player_pieces = player_pieces.transpose()
    run_lengths = [player_pieces[:, i:i + l_len].sum(axis=1) for i in range(len(player_pieces) - l_len + 2)]
    positions = np.where(np.array(run_lengths) == no_pieces)
    if positions[0].size == 0:
        return []

    if not transpose:
        return list(zip(positions[1], positions[0]))
    else:
        return list(zip(positions[0], positions[1]))


def diagonal_lines(player_pieces, l_len, no_pieces):
    results = []
    for i in range(len(player_pieces) - l_len + 1):
        for j in range(len(player_pieces[0]) - l_len + 1):
            if sum(player_pieces[i + x][j + x] for x in range(l_len)) == no_pieces:
                results.append(((i, j), 1))
        for j in range(l_len - 1, len(player_pieces[0])):
            if sum(player_pieces[i + x][j - x] for x in range(l_len)) == no_pieces:
                results.append(((i + 3, j - 3), -1))
    return results


def is_isolated(board, pos):
    i, j = pos
    el = board[i][j]
    b = np.pad(board, pad_width=1, mode='constant', constant_values=2)
    b[i + 1][j + 1] = 3

    return np.all(b[i:i + 3, j:j + 3] != el)


h = [[3, 4, 5, 3, 5, 4, 3],
     [4, 6, 8, 10, 8, 6, 4],
     [5, 8, 11, 13, 11, 8, 5],
     [5, 8, 11, 13, 11, 8, 5],
     [4, 6, 8, 10, 8, 6, 4],
     [3, 4, 5, 7, 5, 4, 3]]
"""
h = [[1, 1,  1,  3,  1, 1, 1],
     [1, 1,  1, 10,  1, 1, 1],
     [1, 1,  1, 13,  1, 1, 1],
     [1, 1,  1, 13,  1, 1, 1],
     [1, 1,  1, 10,  1, 1, 1],
     [1, 1,  1,  7,  1, 1, 1]]
"""


def heuristic2(board):
    probs = heuristic2_prob(board)
    res = probs == np.max(probs)
    return res / res.sum()


@njit
def heuristic2_prob(board):
    h = [[3, 4, 5, 1, 5, 4, 3],
         [4, 6, 8, 13, 8, 6, 4],
         [5, 8, 11, 25, 11, 8, 5],
         [5, 8, 11, 25, 11, 8, 5],
         [4, 6, 8, 25, 8, 6, 4],
         [3, 4, 5, 25, 5, 4, 3]]

    fields = last_nonzero(board)
    prob = np.zeros(fields.size)
    for i, j in enumerate(fields):
        if j != -1:
            prob[i] = h[j][i]

    return prob / np.linalg.norm(prob, 1)


@njit
def last_nonzero(arr):
    res = np.ones(arr.shape[1], dtype=np.int_) * (arr.shape[0] - 1)
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if arr[i][j] != 0:
                res[j] = i - 1
                break
    return res


"""
def last_nonzero(arr):
    mask = arr == 0
    val = arr.shape[0] - np.flip(mask, axis=0).argmax(axis=0) - 1
    return np.where(mask.any(axis=0), val, None)
"""


def heuristic3(cannonical_board):
    # we can win the game so we should

    res = winnable_move(cannonical_board)
    if res is not None:
        return res

    # we can prevent opponent from winning the game so we should play a move to prevent it
    cannonical_board_opponent = cannonical_board * -1
    res = winnable_move(cannonical_board_opponent)
    if res is not None:
        return res

    # we mask away the moves that would make us lose
    mask = np.ones(7)
    valid_moves = cannonical_board_opponent[1] == 0
    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(cannonical_board_opponent[:, j] == 0)
            played_row = available_idx[-1]
            played_row -= 1
            cannonical_board_opponent[played_row][j] = 1
            if connected_four(cannonical_board_opponent, j, played_row):
                mask[j] = 0
            cannonical_board_opponent[played_row][j] = 0

    res = heuristic2_prob(cannonical_board)
    if (res * mask).sum() == 0:
        return res

    return res * mask


def winnable_move(cannonical_board):
    valid_moves = cannonical_board[0] == 0
    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(cannonical_board[:, j] == 0)
            played_row = available_idx[-1]
            cannonical_board[played_row][j] = 1

            if connected_four(cannonical_board, j, played_row):
                res = np.zeros(7)
                res[j] = 1
                cannonical_board[played_row][j] = 0
                return res
            cannonical_board[played_row][j] = 0
    return None


@njit
def connected_four(cannonical_board, played_col, played_row):
    column = cannonical_board[:, played_col]
    for i in range(3):
        if column[i:i + 4].sum() == 4:
            return True

    row = cannonical_board[played_row, :]
    for i in range(4):
        if row[i:i + 4].sum() == 4:
            return True

    start_diag_1 = (played_row - min([played_row, played_col]), played_col - min([played_row, played_col]))
    length_diag_1 = min([6 - start_diag_1[0], 7 - start_diag_1[1]])

    if length_diag_1 >= 4:

        diag_1 = np.array([cannonical_board[start_diag_1[0] + i, start_diag_1[1] + i] for i in range(length_diag_1)])
        for i in range(length_diag_1 - 3):
            if np.all(diag_1[i:i + 4]):
                return True

    start_diag_2 = (played_row + min([5 - played_row, played_col]), played_col - min([5 - played_row, played_col]))
    length_diag_2 = min([start_diag_2[0] + 1, 7 - start_diag_2[1]])

    if length_diag_2 >= 4:
        diag_2 = np.array([cannonical_board[start_diag_2[0] - i, start_diag_2[1] + i] for i in range(length_diag_2)])
        for i in range(length_diag_2 - 3):
            if np.all(diag_2[i:i + 4]):
                return True

    return False


if __name__ == "__main__":

    import time

    board = np.array([[-1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0],
                      [-1, 0, 0, -1, 0, 0, 0],
                      [1, 0, 0, 1, 0, 0, 0]]
                     )

    """
        t = time.time()
        for i in range(repeat):
            last_nonzero2(board)
        print(time.time() - t)
    
        print(last_nonzero(board))
        print(last_nonzero2(board))
    """

    """
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]
                     )
    
    """

    # print(diagonal_lines(board, 4, 3))
    # print(straight_lines(board, 4, 3))
    # print(board_score(board, 1))
