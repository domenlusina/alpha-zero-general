import warnings

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utils import dotdict

warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':
    g = Connect4Game()

    folder = 'C:\\Magistrsko_delo\\alpha-zero-general\\h3\\mcts_visits_tanh_10_max\\default\\'  #'H:\\alpha-zero-trained\\h2\\mcts_heur\\'
    # 'random player', '1-lookahead player',
    enemies = ['engine player'] # 'random player', '1-lookahead player',
    xlabel = 'Weight of heuristic probability '
    # enemy = 'random player'
    # enemy = '1-lookahead player'
    # enemy = 'engine player'
    # enemy = 'alpha-beta player 6'

    rp = RandomPlayer(g).play
    osp = OneStepLookaheadConnect4Player(g).play
    ep = EngineConnect4Player(g).play

    for enemy in enemies:
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        subfolders = sorted(subfolders, key=lambda x: x.split('\\')[-1])
        print(subfolders)
        if enemy == 'random player':
            op = rp
        elif enemy == '1-lookahead player':
            op = osp
        elif enemy == 'engine player':
            op = ep

        report = []
        results = []
        games = 100
        for i, subfolder in enumerate(subfolders):
            print('Progress', i / len(subfolders))
            nn = NNet(g)
            nn.load_checkpoint(subfolder, 'best.pth.tar')
            args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
            mcts1 = MCTS(g, nn, args)
            n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

            arena = Arena.Arena(n1p, op, g, display=display)
            if True:  # enemy == 'engine player':
                result, gamesMoveHistory = arena.playGames(games, verbose=False, returnGamesMoveHistory=True)
                # lists to string format with +1 to move index
                gamesMoveHistoryString = ['\t' + ''.join(map(lambda move: str(move + 1), moves)) + ' ' + str(outcome)
                                          for
                                          moves, outcome in gamesMoveHistory]

                gamesMoveHistoryString.insert(int(len(gamesMoveHistoryString) / 2), '####### switching players')
                print(gamesMoveHistoryString)
            else:
                result = arena.playGames(games, verbose=False)
            results.append(result)
            report.append(str(result) + ' ' + subfolder.split('\\')[-1])

            report.extend(gamesMoveHistoryString)
            report.append("")

        with open(folder + enemy + '.txt', 'w+') as f:
            for line in report:
                f.write(line + '\n')
        """"
        with open(folder + enemy + '.txt', 'r') as f:
            line = f.readline()
            report = ast.literal_eval(line)
        """
        print(results)
        # display_graph(results, [int(subfolder.split('\\')[-1])/100 for subfolder in subfolders], save_path=folder + enemy,
        #             title='Playing against ' + enemy, display=False, xlabel=xlabel, xlim=1.1, xstep=0.1, colwidth=0.07)
