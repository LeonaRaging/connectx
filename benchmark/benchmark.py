import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import subprocess
import time
from src.alpha_beta_bitsboard_minimax_iterative_deepening_null_window import alpha_beta_bitsboard_minimax_iterative_deepening_null_window

TEST_PATHS = ["Test_L1_R1", "Test_L1_R2", "Test_L1_R3", "Test_L2_R1", "Test_L2_R2", "Test_L3_R1"]
NUMBER_OF_TEST = 1000

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

CURRENT_DIR = Path(__file__).resolve().parent

solver = subprocess.Popen(
    [str(CURRENT_DIR / "c4solver"), "-a"],
    cwd=str(CURRENT_DIR),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print(solver.stdout.readline().strip())

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

def get_solver_scores(position):
    print(position)
    solver.stdin.write(position + "\n")
    solver.stdin.flush()

    line = solver.stdout.readline().strip()

    parts = line.split()

    scores = list(map(int, parts[1:]))

    return scores

with open(str(CURRENT_DIR) + "/benchmark_results/benchmark.txt", "w") as f:
    f.write(f"{'Test Set':<15} {'Accuracy':<12} {'Tests Run':<10} {'Time':<8}\n")
    f.write("-" * 50)
    f.write("\n")
for TEST_PATH in TEST_PATHS:
    with open(str(CURRENT_DIR) + "/data_set/" + TEST_PATH) as f:
        correct_moves = 0
        start_time = time.time()
        current_test = 0
        for i, line in enumerate(f):
            position, expected = line.split()

            observation = build_position(position)

            my_move = alpha_beta_bitsboard_minimax_iterative_deepening_null_window(
                observation,
                configuration
            )

            scores = get_solver_scores(position)

            best_move = max(range(7), key=lambda c: scores[c])

            if my_move == best_move:
                correct_moves += 1

            current_test += 1

            if (time.time() - start_time >= 60) or (current_test == NUMBER_OF_TEST):
                break
        with open(str(CURRENT_DIR) + "/benchmark_results/benchmark.txt", "a") as f:
            accuracy = 100 * correct_moves / current_test
            elapsed = time.time() - start_time

            row = (
                f"{TEST_PATH:<15} "
                f"{accuracy:>7.2f}%    "
                f"{current_test:<10} "
                f"{elapsed:>6.2f}s"
            )

            f.write(row + "\n")



