import os
import sys
import time
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np

from Arena import Arena
from MCTS import MCTS
from connect4.Connect4BoardEvaluate import getBoardScore
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Heuristics import heuristic2_prob
from connect4.Connect4Openings import *
from connect4.Connect4Players import EngineConnect4Player
from pytorch_classification.utils import Bar, AverageMeter


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.window_size = args.numItersForTrainExamplesHistoryStart
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        moveHistory = []

        use_opening = random.random() > 0.5
        if use_opening:
            opening = opening_tree()

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            valids = self.game.getValidMoves(canonicalBoard, self.curPlayer)
            pi = pi * valids
            pi = pi / sum(pi)

            if not use_opening or episodeStep >= len(opening):
                if self.args.heuristic_type == 'combined':
                    fraction = self.args.heuristic_probability
                    h_prob = heuristic2_prob(canonicalBoard)
                    new_pi = (np.array(pi) * (1 - fraction) + h_prob * fraction)
                    if self.args.change_probabilities:
                        pi = new_pi

                    action = np.random.choice(len(new_pi), p=new_pi)
                elif self.args.heuristic_type == 'cooling':
                    prob = self.args.heuristic_probability - (episodeStep - 1) * self.args.heuristic_probability / 42
                    if np.random.ranf(1)[0] > prob:
                        action = np.random.choice(len(pi), p=pi)
                    else:
                        action = self.args.heuristic_function(canonicalBoard)
                elif self.args.heuristic_type == 'normal':
                    prob = self.args.heuristic_probability
                    if np.random.ranf(1)[0] > prob:
                        action = np.random.choice(len(pi), p=pi)
                    else:
                        action = self.args.heuristic_function(canonicalBoard)  # , feature=1, pi=pi)
                    if action is None:
                        action = np.random.choice(len(pi), p=pi)

                elif self.args.heuristic_type == 'custom':
                    prob = self.args.probability_function(episodeStep)
                    if np.random.ranf(1)[0] > prob:
                        action = np.random.choice(len(pi), p=pi)
                    else:
                        action = self.args.heuristic_function(canonicalBoard)
                elif self.args.heuristic_type == 'perfect':
                    action = EngineConnect4Player(Connect4Game()).play(canonicalBoard)
                elif self.args.heuristic_type == 'default':
                    action = np.random.choice(len(pi), p=pi)
                else:
                    raise NameError("Wrong heuristic type '" + self.args.heuristic_type + "'")

            else:
                print(opening, episodeStep-1)
                action = opening[episodeStep - 1]
                # pi = np.array(7)
                # pi[action] = 1

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                if np.all(b == canonicalBoard):
                    trainExamples.append([b, self.curPlayer, p, list(moveHistory), None])
                else:
                    trainExamples.append([b, self.curPlayer, p, [6 - x for x in moveHistory], None])

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            moveHistory.append(action)
            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                if self.args.supervised:
                    result = []
                    for x in trainExamples:
                        r = getBoardScore(x[3])
                        result.append((x[0], x[2], r))

                        print(x[0], "Moves", "".join([str(i + 1) for i in x[3]]), "Theoretical value", r)

                    return result
                else:
                    if self.args.value_game_length:
                        r = 1.198 - 99 / 3500 * episodeStep

                    res = [[x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))] for x in trainExamples]
                    if use_opening:
                        for i in range(len(opening)):
                            res[i][2] = 1
                    return res

    def learn(self, verbose=False):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        start_idx = 1
        if self.args.load_model:
            start_idx += self.args.checkpoint_index

        for i in range(start_idx, self.args.numIters + 1):
            self.args.curIter = i

            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = []

                eps_time = AverageMeter()
                if verbose:
                    bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples.extend(self.executeEpisode())

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    if verbose:
                        bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                            total=bar.elapsed_td, eta=bar.eta_td)
                        bar.next()
                if verbose:
                    bar.finish()

                # save the iteration examples to the history

                _, ind = np.unique(np.array(list(map(str, iterationTrainExamples))),
                                   return_index=True)  # removing of duplicates
                self.trainExamplesHistory.append(np.array(iterationTrainExamples)[ind])

            if self.window_size < self.args.numItersForTrainExamplesHistoryMax and \
                    self.args.numItersForTrainExamplesHistoryStart < len(self.trainExamplesHistory):
                self.window_size += 0.5

            if len(self.trainExamplesHistory) > self.window_size:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            # if self.args.heuristic_probability_cooling and self.args.heuristic_probability - self.args.heuristic_probability_cooling_step >= 0:
            #    self.args.heuristic_probability -= self.args.heuristic_probability_cooling_step

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"

        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
