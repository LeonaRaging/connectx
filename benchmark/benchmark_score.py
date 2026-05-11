import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from src.best import best

CURRENT_DIR = Path(__file__).resolve().parent

TEST_PATH = "Test_L1_R2"

class Observation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class Configuration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

configuration = Configuration(7, 6, 4)

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

with open(str(CURRENT_DIR) + "/data_set/" + TEST_PATH) as f:
    correct = 0
    start_time = time.time()
    total = 0
    for i, line in enumerate(f):
        position, expected = line.split()

        if (len(position)) < 14:
            continue

        observation = build_position(position)

        start_time = time.time()

        score = best(observation, configuration)

        print(time.time() - start_time)

        if score == int(expected):
            correct += 1
        # else:
        print(position + ' ' + str(score) + ' ' + expected)

        # break
        total = i + 1

        # if time.time() - start_time > 2:
        #     break
    print(100 * correct / total)
    print(total)