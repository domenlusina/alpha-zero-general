import yaml
from CoachHeuristic import Coach
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Heuristics import heuristic1, heuristic2
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict

with open(".training_config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["CUDA_VISIBLE_DEVICES"])
import tensorflow as tf

folder = './{}_{}/{}'.format(cfg["heuristic_function"], cfg["heuristic_type"], cfg["heuristic_probability"])
cp_idx = 0

if cfg["heuristic_function"] == "h1":
    heuristic_function = heuristic1
elif cfg["heuristic_function"] == "h2":
    heuristic_function = heuristic2
else:
    raise Exception('Unknown heuristic function {}.'.format(cfg["heuristic_function"]))

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

    'heuristic_probability': cfg["heuristic_probability"]/100,
    'heuristic_type': cfg["heuristic_type"],
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
        print(args)
        g = Connect4Game()
        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()
