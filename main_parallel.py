from CoachHeuristicParallel import Coach
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Heuristics import heuristic1
from utilities import dotdict
import os

folder = './temp_h1_50_parallel'

args = dotdict({
    'numIters': 100,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.6,  # During arena playoff, new neural net will be accepted if threshold is surpasses.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': folder,
    'numItersForTrainExamplesHistory': 20,

    'heuristic_probability': 0.50,
    'heuristic_type': 'normal',
    'heuristic_function': heuristic1,

    'multiGPU': False,
    'setGPU': '0',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 4,
    'numPerProcessSelfPlay': 10,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 4,
    'numPerProcessAgainst': 10,
})

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    g = Connect4Game()
    c = Coach(g, args)
    c.learn()
