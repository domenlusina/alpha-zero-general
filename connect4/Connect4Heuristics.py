import numpy as np
from numba import njit, i1, b1, int64


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


def heuristic1(board, pi=np.ones(7) / 7, feature=4):
    player = 1
    # if np.sum(board) == 0:
    #     player = 1
    return heuristic1player(board, player, pi, feature)


def heuristic1player(board, player, pi, feature):
    valid_moves = board[0] == 0
    scores = np.zeros(valid_moves.size)
    valid_moves_scores = []
    for j in range(valid_moves.size):
        if valid_moves[j]:
            available_idx, = np.where(board[:, j] == 0)
            board[available_idx[-1]][j] = player
            scores[j] = board_score(board, player, feature)
            if scores[j] != np.inf:
                scores[j] -= board_score(board, -player, feature)
            board[available_idx[-1]][j] = 0  # we undo the move
            valid_moves_scores.append(scores[j])
        else:
            scores[j] = -np.inf

    if len(set(valid_moves_scores)) == 1:
        pi = valid_moves * pi
        pi /= sum(pi)

        return np.random.choice(len(pi), p=pi)

    return np.argmax(scores)


def board_score(board, player, feature):
    board = np.copy(board)
    score = 0
    player_pieces = board * player
    shape = board.shape
    l_len = 4

    """FEATURE 1"""
    if feature >= 1:
        f1l = straight_lines(player_pieces, l_len, 4)
        f1c = straight_lines(player_pieces, l_len, 4, transpose=True)
        f1d = diagonal_lines(player_pieces, l_len, 4)

        if f1l or f1c or f1d:
            return np.inf

    """FEATURE 2"""
    if feature >= 2:
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

    if feature >= 2:
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

    if feature >= 3:
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

    if feature >= 4:
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
    fields = last_nonzero(board)
    scores = [h[j][i] if j is not None else 0 for i, j in enumerate(fields)]

    idx = np.array([i for i, score in enumerate(scores) if max(scores) == score])

    return np.random.choice(idx)


def heuristic2_prob_max(board):
    fields = last_nonzero(board)
    scores = [h[j][i] if j is not None else 0 for i, j in enumerate(fields)]

    res = np.array([max(scores) == score for i, score in enumerate(scores)])
    return res / res.sum()


def heuristic2_array(board):
    fields = last_nonzero(board)
    fields = [h[j][i] if j is not None else 0 for i, j in enumerate(fields)]
    return fields


def heuristic2_prob(board):
    """"
    h = [[3, 4, 5, 1, 5, 4, 3],
         [4, 6, 8, 13, 8, 6, 4],
         [5, 8, 11, 25, 11, 8, 5],
         [5, 8, 11, 25, 11, 8, 5],
         [4, 6, 8, 25, 8, 6, 4],
         [3, 4, 5, 25, 5, 4, 3]]
    """
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
    length_diag_2 = min([start_diag_2[0]+1, 7 - start_diag_2[1]])

    if length_diag_2 >= 4:
        diag_2 = np.array([cannonical_board[start_diag_2[0] - i, start_diag_2[1] + i] for i in range(length_diag_2)])
        for i in range(length_diag_2 - 3):
            if np.all(diag_2[i:i + 4]):
                return True

    return False


""""
    # major threat
    major_threats = []
    major_threats.extend(major_threats_line(board, transpose=False))
    major_threats.extend(major_threats_line(board, transpose=True))
    major_threats.extend(major_threats_diag(board, transpose=False))
    major_threats.extend(major_threats_diag(board, transpose=True))

    major_threats_field = np.zeros((6, 7))

    for x, y in major_threats:
        major_threats_field[x, y] = 1

    for i in range(7):
        found_odd = False
        for j in range(5, -1, -2):
            if found_odd:
                major_threats_field[j, i] = 0
            if major_threats_field[j, i] == 1:
                found_odd = True

        found_even = False
        for j in range(4, -1, -2):
            if found_even:
                major_threats_field[j, i] = 0
            if major_threats_field[j, i] == 1:
                found_even = True

    # minor threats
    
    minor_threats = []
    minor_threats.extend(filter_minor_threats(minor_threats_line(board, transpose=False), major_threats))
    minor_threats.extend(filter_minor_threats(minor_threats_line(board, transpose=True), major_threats))
    minor_threats.extend(filter_minor_threats(minor_threats_diag(board, transpose=False), major_threats))
    minor_threats.extend(filter_minor_threats(minor_threats_diag(board, transpose=True), major_threats))
    """


def filter_minor_threats(minor_threats, major_threats):
    res = []
    for i in range(0, len(minor_threats), 2):
        if (not minor_threats[i] in major_threats) and (not minor_threats[i + 1] in major_threats):
            res.append(minor_threats[i])
            res.append(minor_threats[i + 1])
    return res


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


@njit
def major_threats_diag(board, transpose=False):
    if transpose:
        board = board.transpose()

    threats = []

    for i in range(board.shape[0] - 3):
        for j in range(board.shape[1] - 3):
            tmp = np.array([board[i, j], board[i + 1, j + 1], board[i + 2, j + 2], board[i + 3, j + 3]])
            if tmp.sum() == 3:
                offset = np.argmin(tmp)
                threats.append((i + offset, j + offset))

    if transpose:
        threats = [(y, x) for x, y in threats]

    return threats


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


if __name__ == "__main__":
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0]]
                     )

    print(heuristic2_array(board))
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
