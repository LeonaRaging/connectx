from src.main import main

import numpy as np

class Observation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class Configuration:
    def __init__(self, columns, rows, inarow, timeout):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.timeout = timeout

configuration = Configuration(7, 6, 4, 2)

def build_position(move_string):
    observation = Observation([0] * 42, 1)
    heights = [0] * 7

    for c in move_string:
        col = int(c) - 1
        row = heights[col]

        observation.board[(5 - row) * 7 + col] = observation.mark

        heights[col] += 1
        observation.mark = 3 - observation.mark

    return observation

def print_board(board):
    grid = np.array(board).reshape(6, 7)

    for r in range(6):
        for c in range(7):

            piece = grid[r][c]

            if piece == 0:
                print(".", end=" ")
            elif piece == 1:
                print("X", end=" ")
            else:
                print("O", end=" ")

        print()

    print("1 2 3 4 5 6 7")
    print()

position = ""

while True:
    c = input()
    position += str(c)
    
    observation = build_position(position)

    print_board(observation.board)
    print(position)

    my_move = main(observation, configuration)

    print(my_move + 1)

    position += str(my_move + 1)

    observation = build_position(position)

    print_board(observation.board)
    print(position)


