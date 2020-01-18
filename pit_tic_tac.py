from utilities import *

import Arena
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as NNet

"""
use this script to play any two agents against each utilities, or play manually with
any agent.
"""

g = TicTacToeGame()

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g).play
op = OptimalPlayer(g).play

# nnet players

n1 = NNet(g)
n1.load_checkpoint('./temp_tic_tac_toe/', 'checkpoint_1.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# n2 = NNet(g)
# n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
# args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
# mcts2 = MCTS(g, n2, args2)
# n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(op, n1p, g, display=display)
oneWon, twoWon, draws = arena.playGames(100, verbose=False)
print()
print("Player1 won " + str(oneWon) + " games.")
print()
print("Player2 won " + str(twoWon) + " games.")
print()
print("There were " +str(draws) + " draws.")

