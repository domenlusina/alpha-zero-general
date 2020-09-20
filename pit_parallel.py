import warnings
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

import Arena
from GraphDrawing import display_graph
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict

warnings.filterwarnings('ignore', category=FutureWarning)


def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.index' in file:
                if 'best' not in file and 'temp' not in file:
                    files.append(file[:-6])
    return files


"""
Functions used for parallel play
"""


def player_vs_nnet(results_folder, player, games, chunk):
    g = Connect4Game()
    results = []
    chunk_number, files = chunk

    for file in files:
        nn = NNet(g)
        nn.load_checkpoint(results_folder, file)
        args = dotdict({'numMCTSSims': 25, 'cpuct': 5.0})
        mcts1 = MCTS(g, nn, args)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, player, g, display=display)
        result = arena.playGames(games, verbose=False)
        results.append(list(result))

    return chunk_number, results


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parallel_play(files, results_folder, player, games=100, workers=10):
    chunks = list(divide_chunks(files, int(len(files) / workers)))
    numbered_chunks = [(i, chunk) for i, chunk in enumerate(chunks)]
    pool = Pool(processes=workers)

    chunk_res = pool.map(partial(player_vs_nnet, results_folder, player, games), numbered_chunks)
    pool.close()
    pool.join()
    chunk_res1 = list(chunk_res)
    chunks_res1 = sorted(chunk_res1, key=lambda x: x[0])

    results = []
    for cr in chunks_res1:
        results.extend(cr[1])

    return results


if __name__ == '__main__':
    g = Connect4Game()
    # all players
    rp = RandomPlayer(g).play
    osp = OneStepLookaheadConnect4Player(g).play
    ep = EngineConnect4Player(g).play
    games = 100
    workers = 10

    results_folders = ['H:\\alpha-zero-trained\\h0\\15blocks']
    for results_folder in results_folders:
        files = get_files(results_folder)
        files = sorted(files, key=lambda file: int(file[file.find('_') + 1: file.find('.')]))
        ind = [int(file[file.find('_') + 1: file.find('.')]) for file in files]

        rand_scores = parallel_play(files, results_folder, rp, games=games, workers=workers)
        print("Random player done")
        print(rand_scores)
        display_graph(rand_scores, ind, save_path=results_folder + 'random.png', title='Playing against random player')

        ahead_scores = parallel_play(files, results_folder, osp, games=games, workers=workers)
        print("One move ahead player done")
        print(ahead_scores)
        display_graph(ahead_scores, ind, save_path=results_folder + '1ahead.png',
                      title='Playing against 1-lookahead player')

        engine_scores = parallel_play(files, results_folder, ep, games=games, workers=workers)
        print("Engine player done")
        print(engine_scores)
        display_graph(engine_scores, ind, save_path=results_folder + 'engine.png', title='Playing against engine')

        with open(results_folder + 'results.txt', "w+") as f:
            f.write(str(rand_scores) + '\n')
            f.write(str(ahead_scores) + '\n')
            f.write(str(engine_scores) + '\n')
