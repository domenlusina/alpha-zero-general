import tensorflow as tf

from CoachHeuristic import Coach
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Heuristics import *
from connect4.tensorflow.NNet import NNetWrapper as nn
from utilities import dotdict

folder = './temp_h1_50/'
cp_idx = 0

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
    'load_model': False,
    'checkpoint_index': cp_idx,
    'load_folder_file': (folder, 'checkpoint_' + str(cp_idx) + '.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'heuristic_probability': 0.5,
    'heuristic_type': 'normal',
    'heuristic_function': heuristic1
})

if __name__ == "__main__":
    run = False
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        run = True
    else:
        print("Please install GPU version of TF")

    if run:
        g = Connect4Game()
        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()
