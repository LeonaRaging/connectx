from alpha_beta_bitsboard_minimax_iterative_deepening_null_window import alpha_beta_bitsboard_minimax_iterative_deepening_null_window
from best import best

def main(observation, configuration):
    moves = sum(1 for x in observation.board if x != 0)
    if moves < 18:
        return alpha_beta_bitsboard_minimax_iterative_deepening_null_window(observation, configuration)
    else:
        return best(observation, configuration)