# Alpha Zero General

This project is based on the [Alpha Zero General](https://github.com/suragnair/alpha-zero-general) implementation. We experimented with the idea of using domain knowledge to improve learning for Connect Four. We introduced domain knowledge in the form of an expert heuristic. The expert heuristic was only used in the learning phase of the AlphaZero algorithm.

## Heuristics used

We used the Connect Four heuristics described in [article] (https://www.researchgate.net/publication/331552609_Research_on_Different_Heuristics_for_Minimax_Algorithm_Insight_from_Connect-4_Game). The article introduces two heuristics, which we called Field Heuristic (Heuristic-1) and Feature Heuristic (Heuristic-2). The idea behind the field heuristic is that not all fields of the map are equal. Some that are closer to the center of the board are generally better because they provide more opportunities to connect four tiles. The field heuristic encourages connecting your own tokens and prevents your opponent from connecting their tokens. The implementation of the heuristic can be seen [here] (alpha-zero-general/connect4/Connect4Heuristics.py).

## Mehtods of using heuristics
We have experimented with two ways of using domain knowledge in learning. One is used during the self-play and the other helps guide the Monte Carlo tree search.
In self-play, we introduced methods that decide the next move with a certain probability according to the heuristic used. When guiding the MCTS heuristics, the UCB formula is modified such that for a given board state s and an action a, the probability P(s, a) also contains the move recommended by the heuristic. Let P'(s, a) be a new probability of a move for a given state that also includes heuristics. We define P'(s, a)=1-f(x)*P(s,a)+f(x)*H(s), where f(x) is a weight function that decides how much we trust our heuristic, and x is the number of visits, P(s,a) is determined by the neural network, and H(s) returns the move for state s recommended by the heuristic.

The project also includes several ways to evaluate (playing against players with different difficulty levels, playing with error correction, sets of positions ) and different metrics to evaluate the results. 

If you are interested in this project and want to know more, drop me an email.




