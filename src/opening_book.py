import numpy as np

ROWS = 6
COLS = 7

# Source - https://stackoverflow.com/a/74950580
# Posted by trincot, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-11, License - CC BY-SA 4.0

def getstacks(board):
    counts = [0, 0, 0]
    # Convert data structure to stacks -- one stack per column
    stacks = [[] for _ in board[0]]
    for row, values in enumerate(reversed(board)):
        for col, (value, stack) in enumerate(zip(values, stacks)):
            if value:
                # Verify there are no holes
                if len(stack) != row:
                    raise ValueError(f"The disc at {row+1},{col+1} should not be floating above an empty cell")
                stack.append(value)
                counts[value] += 1
    if not (0 <= counts[1] - counts[2] <= 1):
        raise ValueError("Number of discs per player is inconsistent")
    return stacks, 1 + counts[1] - counts[2] 


def searchmoves(stacks, player):
    # Perform a depth first search with backtracking
    for col, stack in enumerate(stacks):
        if stack and stack[-1] == player:
            stack.pop()
            moves = searchmoves(stacks, 3 - player)
            stack.append(player)  # Restore
            if moves is not None:  # Success
                moves.append(col + 1)
                return moves
    if any(stacks):
        return None  # Stuck: backtrack.
    return []  # Success: all discs were removed


def solve(board):
    stacks, nextplayer = getstacks(board)
    return searchmoves(stacks, 3 - nextplayer)


book = {}

with open("book.txt") as f:
    for line in f:
        pos, move = line.split()
        book[pos] = int(move)

def opening_book(observation, configuration):
    grid = np.array(observation.board).reshape(ROWS, COLS)

    position = "".join(map(str, solve(grid) or []))

    if position == "":
        position = "0"

    move = book.get(position)

    if move is None:
        return None

    return move - 1