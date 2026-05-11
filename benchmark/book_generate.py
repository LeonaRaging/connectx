import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import subprocess

CURRENT_DIR = Path(__file__).resolve().parent
order = [3, 2, 4, 1, 5, 0, 6]

solver = subprocess.Popen(
    [str(CURRENT_DIR / "c4solver"), "-a"],
    cwd=str(CURRENT_DIR),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print(solver.stdout.readline().strip())


def get_solver_scores(position):
    solver.stdin.write(position + "\n")
    solver.stdin.flush()

    line = solver.stdout.readline().strip()

    parts = line.split()

    # print(parts)

    if "Invalid" in line:
        return None

    if position == "":
        scores = list(map(int, parts))
    else:
        scores = list(map(int, parts[1:]))

    return scores


def run(depth):
    position = ''.join(str(m + 1) for m in moves[:depth])

    scores = get_solver_scores(position)

    if scores == None:
        return None

    best_score = -100
    best_move = -1
    for i in range(7):
        if scores[order[i]] > best_score:
            best_score = scores[order[i]]
            best_move = order[i]

    with open("book.txt", "a") as f:
        f.write(position + ' ' + str(best_move + 1) + '\n')


height = [0] * 7
moves = [0] * 42

def Try(depth):
    if depth > 7:
        return
    run(depth)


    for col in range(7):
        if height[col] < 6:
            moves[depth] = col

            height[col] += 1

            Try(depth + 1)

            height[col] -= 1


Try(0)

solver.terminate()