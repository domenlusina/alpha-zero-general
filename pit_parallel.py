import os
import warnings
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utilities import dotdict

warnings.filterwarnings('ignore', category=FutureWarning)

"""
Functions used to draw graphs
"""


def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.index' in file:
                if 'best' not in file and 'temp' not in file:
                    files.append(file[:-6])
    return files


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def display_graph(data, ind, save_path=None, title='Title', display=False):
    data = np.array(data)
    wins = data[:, 0]
    loses = data[:, 1]
    draws = data[:, 2]

    c = ['#396AB1', '#DA7C30', '#3E9651', '#CC2529']  # our color palette
    c1 = []
    c2 = []
    c3 = []
    for i, j in enumerate(ind):
        v = 1.375 - 0.0075 * j
        c1.append(lighten_color(c[0], v))
        c2.append(lighten_color(c[1], v))
        c3.append(lighten_color(c[2], v))

    # drawing stacked bars
    width = 0.7
    b1 = plt.bar(ind, wins, width=width, color=c1)
    b2 = plt.bar(ind, draws, width=width, bottom=wins, color=c2)
    b3 = plt.bar(ind, loses, width=width, bottom=draws + wins, color=c3)

    # drawing line
    middle = wins + draws / 2
    plt.plot(ind, middle, 'o-', color=c[3], linewidth=1, markersize=2)

    # writings
    plt.ylabel('Percent of games [%]')
    plt.xlabel('Episodes of self-play')
    plt.title(title)

    # legend
    leg = plt.legend((b1[0], b2[0], b3[0]), ('Wins', 'Draws', 'Loses'), loc='upper right')
    for i in range(3):
        leg.legendHandles[i].set_color(c[i])

    # limit axis
    plt.xlim(0, 101)
    plt.ylim(0, 101)

    # ticks
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(np.arange(0, 101, 10), rotation=90)
    aml = AutoMinorLocator(10)
    plt.axes().xaxis.set_minor_locator(aml)
    plt.axes().yaxis.set_minor_locator(aml)

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.clf()


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
        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
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

    results_folders = ['.\\temp_h1_50\\']
    for results_folder in results_folders:

        files = get_files(results_folder)
        files = sorted(files, key=lambda file: int(file[file.find('_') + 1: file.find('.')]))
        ind = [int(file[file.find('_') + 1: file.find('.')]) for file in files]

        save_folder = '.\\graphs\\' + results_folder[results_folder.find('_') + 1:]
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        rand_scores = parallel_play(files, results_folder, rp, games=games, workers=workers)
        print("Random player done")
        print(rand_scores)
        display_graph(rand_scores, ind, save_path=save_folder + 'random.png', title='Playing against random player')

        ahead_scores = parallel_play(files, results_folder, osp, games=games, workers=workers)
        print("One move ahead player done")
        print(ahead_scores)
        display_graph(ahead_scores, ind, save_path=save_folder + '1ahead.png',
                      title='Playing against 1-lookahead player')

        engine_scores = parallel_play(files, results_folder, ep, games=games, workers=workers)
        print("Engine player done")
        print(engine_scores)
        display_graph(engine_scores, ind, save_path=save_folder + 'engine.png', title='Playing against engine')

        with open(save_folder + 'results.txt', "w+") as f:
            f.write(str(rand_scores) + '\n')
            f.write(str(ahead_scores) + '\n')
            f.write(str(engine_scores) + '\n')
