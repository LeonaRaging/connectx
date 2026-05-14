from .mcts import mcts
from .best import best
from .opening_book import opening_book

def main(observation, configuration):
    moves = sum(1 for x in observation.board if x != 0)
    if moves < 8:
        return opening_book(observation, configuration)
    elif moves < 17:
        return mcts(observation, configuration)
    else:
        return best(observation, configuration)