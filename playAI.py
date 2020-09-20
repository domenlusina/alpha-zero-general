from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import HumanConnect4Player
from connect4.tensorflows.NNet import NNetWrapper as NNet
from utils import dotdict
import numpy as np

if __name__ == '__main__':
    goingFirst = True
    folder = "H:\\alpha-zero-trained\\final\\h2\\mcts_visits_tanh\\default\\1\\"

    game = Connect4Game()
    nn = NNet(game)
    nn.load_checkpoint(folder, 'best.pth.tar')
    args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
    mcts1 = MCTS(game, nn, args)
    AI = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    human = HumanConnect4Player(game).play

    if goingFirst:
        players = [AI, None, human]
    else:
        players = [human, None, AI]

    curPlayer = 1
    board = game.getInitBoard()
    while game.getGameEnded(board, curPlayer) == 0:
        display(board, symbols=True)

        action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

        while valids[action] == 0:
            print("Move", action, "is illegal. You can play:", [i for i, legal in enumerate(board[0] == 0) if legal])
            action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))

        board, curPlayer = game.getNextState(board, curPlayer, action)

    display(board)
    print("Game over!", "Result ", str(game.getGameEnded(board, 1)))
