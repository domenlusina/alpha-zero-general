import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from MCTS import MCTS
import Arena
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict


def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.index' in file:
                if 'best' not in file and 'temp' not in file:
                    files.append(file[:-6])
    return files


def history2string(gamesMoveHistory):
    gamesMoveHistoryString = ['\t' + ''.join(map(lambda move: str(move + 1), moves)) + ' ' + str(outcome)
                              for
                              moves, outcome in gamesMoveHistory]

    gamesMoveHistoryString.insert(int(len(gamesMoveHistoryString) / 2), '####### switching players')
    return gamesMoveHistoryString


def getReport(scores, files, moves):
    report = ""
    for i, file in enumerate(files):
        report += str(scores[i]) + " " + file + "\n"
        if len(moves) > 0:
            report += "\n".join(moves[i])
            report += "\n"
    return report


if __name__ == '__main__':
    g = Connect4Game()
    # all players
    rp = RandomPlayer(g).play
    osp = OneStepLookaheadConnect4Player(g).play
    ep = EngineConnect4Player(g).play
    # vp = VelenaConnect4Player(g).play

    """
    arena = Arena.Arena(vp, ep, g, display=display)
    results = arena.playGames(100, verbose=False)
    print(results)
    """
    rand_scores = []
    ahead_scores = []
    engine_scores = []
    games = 100

    moves_rand = []
    moves_ahead = []
    moves_engine = []

    results_folder = 'C:\\Magistrsko_delo\\alpha-zero-general\\h2\\basic\\cooling_iter\\50\\'  # 'H:\\alpha-zero-trained\\h0\\cpuct\\1\\'
    files = get_files(results_folder)

    files = sorted(files, key=lambda file: int(file[file.find('_') + 1: file.find('.')]))
    ind = [int(file[file.find('_') + 1: file.find('.')]) for file in files]
    print(files)
    for file in files:
        print("Checkpoint", file)
        # nnet players
        n1 = NNet(g)
        n1.load_checkpoint(results_folder, file)
        args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1})
        mcts1 = MCTS(g, n1, args1)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        """
        arena = Arena.Arena(n1p, rp, g, display=display)
        results, gamesMoveHistory = arena.playGames(games, verbose=False, returnGamesMoveHistory=True)
        rand_scores.append(list(results))
        moves_rand.append(history2string(gamesMoveHistory))

        arena = Arena.Arena(n1p, osp, g, display=display)
        results, gamesMoveHistory = arena.playGames(games, verbose=False, returnGamesMoveHistory=True)
        ahead_scores.append(list(results))
        moves_ahead.append(history2string(gamesMoveHistory))
        """
        arena = Arena.Arena(n1p, ep, g, display=display)
        results, gamesMoveHistory = arena.playGames(games, verbose=False, returnGamesMoveHistory=True)
        engine_scores.append(list(results))
        moves_engine.append(history2string(gamesMoveHistory))

    print(rand_scores)
    print()

    print(ahead_scores)
    print()

    print(engine_scores)
    print()

    """
    display_graph(rand_scores, ind, save_path=results_folder + 'random.png', title='Playing against random player')
    display_graph(ahead_scores, ind, save_path=results_folder + '1ahead.png',
                  title='Playing against 1-lookahead player')
    display_graph(engine_scores, ind, save_path=results_folder + 'engine.png', title='Playing against engine')
    """
    # with open(results_folder + 'random player.txt', "w+") as f:
    #    f.write(getReport(rand_scores, files, moves_rand))

    # with open(results_folder + '1-lookahead player.txt', "w+") as f:
    #    f.write(getReport(ahead_scores, files, moves_ahead))

    with open(results_folder + 'engine player.txt', "w+") as f:
        f.write(getReport(engine_scores, files, moves_engine))

    from connect4.Connect4BoardEvaluate import *

    tmp = analyseEngine(results_folder + "engine player.txt")

    print(" ".join([str(x) for x in ind]))
    print(" ".join([str(x) for x in tmp]))

    with open(results_folder + 'engine player.txt', "a") as f:
        f.write('\n')
        f.write(" ".join([str(x) for x in ind]) + '\n')
        f.write(" ".join([str(x) for x in tmp]) + '\n')
