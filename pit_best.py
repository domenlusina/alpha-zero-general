import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import ast
from MCTS import MCTS
import Arena
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utils import dotdict


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


def display_graph(data, ind, save_path=None, title='Title', display=False, xlabel ='Starting heuristic probability [%]') :
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
    plt.xlabel(xlabel)
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


if __name__ == '__main__':
    g = Connect4Game()
    folder = 'H:\\alpha-zero-trained\\h0\\cpuct\\'
    enemies = ["random player", "1-lookahead player", "engine player"]
    xlabel = 'CPUCT constant'
    # enemy = "random player"
    # enemy = "1-lookahead player"
    # enemy = "engine player"
    # enemy = "alpha-beta player 6"

    rp = RandomPlayer(g).play
    osp = OneStepLookaheadConnect4Player(g).play
    ep = EngineConnect4Player(g).play

    for enemy in enemies:
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        # subfolders = ['H:\\alpha-zero-trained\\h0\\cpuct\\5']
        subfolders = sorted(subfolders, key=lambda x: int(x.split("\\")[-1]))

        if enemy == "random player":
            op = rp
        elif enemy == "1-lookahead player":
            op = osp
        elif enemy == "engine player":
            op = ep

        results = []
        games = 100
        for i, subfolder in enumerate(subfolders):
            print("Progress", i/len(subfolders))
            nn = NNet(g)
            nn.load_checkpoint(subfolder, 'best.pth.tar')
            print("CPUCT",int(subfolder.split("\\")[-1]))
            args = dotdict({'numMCTSSims': 25, 'cpuct': int(subfolder.split("\\")[-1])})
            mcts1 = MCTS(g, nn, args)
            n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

            arena = Arena.Arena(n1p, op, g, display=display)
            result = arena.playGames(games, verbose=False)
            results.append(result)

        print(results)
        with open(folder + enemy + '.txt', "w+") as f:
            f.write(str(results) + '\n')
        """"
        with open(folder + enemy + '.txt', "r") as f:
            line = f.readline()
            results = ast.literal_eval(line)
        """
        display_graph(results, [int(subfolder.split("\\")[-1]) for subfolder in subfolders], save_path=folder + enemy,
                      title='Playing against ' + enemy, display=False, xlabel = xlabel)
