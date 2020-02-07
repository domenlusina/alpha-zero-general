import numpy as np


def heuristic1(board):
    player = -1
    if np.sum(board) == 0:
        player = 1
    return heuristic1player(board, player)


def heuristic1player(board, player):
    valid_moves = board[0] == 0
    scores = np.zeros(valid_moves.size)
    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(board[:, j] == 0)
            board[available_idx[-1]][j] = player
            scores[j] = board_score(board, player)
            scores[j] -= board_score(board, -player)
            board[available_idx[-1]][j] = 0  # we undo the move
        else:
            scores[j] = -np.inf
    return np.argmax(scores)


def board_score(board, player):
    score = 0
    player_pieces = board * player
    shape = board.shape
    l_len = 4

    """FEATURE 1"""
    f1l = straight_lines(player_pieces, l_len, 4)
    f1c = straight_lines(player_pieces, l_len, 4, transpose=True)
    f1d = diagonal_lines(player_pieces, l_len, 4)

    if f1l or f1c or f1d:
        return np.inf

    """FEATURE 2"""
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


def heuristic2(board):
    h = [[3, 4, 5, 7, 5, 7, 3],
         [4, 6, 8, 10, 8, 6, 4],
         [5, 8, 11, 13, 11, 8, 5],
         [5, 8, 11, 13, 11, 8, 5],
         [4, 6, 8, 10, 8, 6, 4],
         [3, 4, 5, 7, 5, 4, 3]]

    fields = last_nonzero(board)
    score = 0
    idx = 0
    for i, j in enumerate(fields):
        if j is not None and score < h[j][i]:
            score = h[j][i]
            idx = i

    return idx


def heuristic2_prob(board):
    h = [[3, 4, 5, 7, 5, 7, 3],
         [4, 6, 8, 10, 8, 6, 4],
         [5, 8, 11, 13, 11, 8, 5],
         [5, 8, 11, 13, 11, 8, 5],
         [4, 6, 8, 10, 8, 6, 4],
         [3, 4, 5, 7, 5, 4, 3]]

    fields = last_nonzero(board)
    prob = np.zeros(fields.size)
    for i, j in enumerate(fields):
        if j is not None:
            prob[i] = h[j][i]

    return prob / np.linalg.norm(prob, 1)


def last_nonzero(arr):
    mask = arr == 0
    val = arr.shape[0] - np.flip(mask, axis=0).argmax(axis=0) - 1
    return np.where(mask.any(axis=0), val, None)


if __name__ == "__main__":
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]
                     )
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
