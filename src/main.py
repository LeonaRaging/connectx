from alpha_beta_bitsboard_minimax import alpha_beta_bitsboard_minimax
from best import best
from opening_book import opening_book

def main(observation, configuration):
    moves = sum(1 for x in observation.board if x != 0)
    if moves < 8:
        return opening_book(observation, configuration)
    elif moves < 17:
        return alpha_beta_bitsboard_minimax(observation, configuration)
    else:
        return best(observation, configuration)