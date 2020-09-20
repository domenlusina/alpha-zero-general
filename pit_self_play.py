import warnings

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict

warnings.filterwarnings('ignore', category=FutureWarning)
if __name__ == '__main__':
    g = Connect4Game()
    model_path = "H:\\alpha-zero-trained\\final\\h2\\basic\\cooling\\20\\"
    games = 100
    nn = NNet(g)

    checkpoints = []

    for r, d, f in os.walk(model_path):
        for file in f:
            if '.index' in file:
                if 'temp' not in file and 'best' not in file:
                    checkpoints.append(file[:-6])
    print(checkpoints)

    for checkpoint in checkpoints:
        print(checkpoint)
        nn.load_checkpoint(model_path, checkpoint)
        args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
        mcts1 = MCTS(g, nn, args)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        arena = Arena.Arena(n1p, n1p, g, display=display)
        result, gamesMoveHistory = arena.playGames(games, verbose=False, returnGamesMoveHistory=True)

        c = checkpoint.split(".pth")
        with open(model_path + c[0]+ "_self_play" + '.txt', 'w+') as f:
            for line in ["".join([str(el + 1) for el in x[0]]) + '\n' for x in gamesMoveHistory]:
                f.write(line)
