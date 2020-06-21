from pathlib import Path

import yaml

from CoachHeuristic import Coach
from ProbFunctions import *
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Heuristics import *
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict

with open(".training_config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(cfg)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["CUDA_VISIBLE_DEVICES"])
import tensorflow as tf

if cfg["heuristic_function"] == "h0" or cfg["heuristic_function"] == "default":
    folder = './h0/' + str(cfg["dirname"])
elif cfg["heuristic_type"] == 'perfect':
    folder = './{}/{}'.format(cfg["heuristic_type"], cfg["dirname"])
elif 'controlled' in cfg["dirname"]:
    folder = './{}/'.format(cfg["dirname"])
else:
    folder = './{}/{}/{}/{}'.format(cfg["heuristic_function"], cfg["dirname"], cfg["heuristic_type"],
                                    cfg["heuristic_probability"])

cp_idx = 0

if cfg["heuristic_function"] == "h0":
    heuristic_function = None
elif cfg["heuristic_function"] == "h1":
    heuristic_function = heuristic1
elif cfg["heuristic_function"] == "h2":
    heuristic_function = heuristic2
elif cfg["heuristic_function"] == "h3":
    heuristic_function = heuristic3
elif cfg["heuristic_function"] == "h2array":
    heuristic_function = heuristic2_array
elif cfg["heuristic_function"] == "h1look":
    heuristic_function = heuristic1lookahead
else:
    raise Exception('Unknown heuristic function {}.'.format(cfg["heuristic_function"]))

args = dotdict({
    'curIter': 0,
    'numIters': 100,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.55,  # During arena playoff, new neural net will be accepted if threshold is surpasses.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': cfg["cpuct"],

    'supervised': False,  # does solver evaluate boards state and give them theoretical values

    'use_dirichlet': False,
    'dirichlet_alpha': 1.75,
    'dirichlet_weight': 0.25,

    'numItersForTrainExamplesHistoryStart': 20,  # when the window will start increasing its size every second iteration
    'numItersForTrainExamplesHistoryMax': 20,  # when the window size stops increasing

    'value_game_length': False,

    'heuristic_function_name': cfg["heuristic_function"],
    'heuristic_probability': cfg["heuristic_probability"] / 100,
    'heuristic_type': cfg["heuristic_type"],
    'heuristic_function': heuristic_function,
    'probability_function': None,  # lambda x: cutoff(x, start_prob=cfg["heuristic_probability"] / 100, cutoffmove=10),

    'mcts_with_heuristics': False,
    'mcts_with_heuristics_visits': '',  # options: tanh, 1/x
    'c': cfg["heuristic_probability"],

    'change_probabilities': False,

    # 'heuristic_probability_cooling': False,
    # 'heuristic_probability_cooling_step': 0.05,

    'checkpoint': folder,
    'load_model': False,
    'checkpoint_index': 0,
    'load_folder_file': (folder, 'checkpoint_' + str(cp_idx) + '.pth.tar'),
})

if __name__ == "__main__":
    run = False
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        run = True
    else:
        print("Please install GPU version of TF")

    if run:
        if not os.path.exists(folder):
            Path(folder).mkdir(parents=True, exist_ok=True)

        with open(folder + '/args.txt', 'w+') as f:
            f.write(str(cfg) + '\n')
            f.write('\n')
            f.write(str(args))
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
