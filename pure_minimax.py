import numpy as np

COLS = None
ROWS = None
board = None
INAROW = None
ALL_WINDOWS = None
player_id = None
opponent_id = None
MAX_DEPTH = 3
INF = 1e9
f = None

# Get every INAROW segment from all 4 directions
def get_all_windows():
    windows = []
    for r in range(ROWS):
        for c in range(COLS - INAROW + 1):
            windows.append([(r, c + i) for i in range(INAROW)])

    for r in range(ROWS - INAROW + 1):
        for c in range(COLS):
            windows.append([(r + i, c) for i in range(INAROW)])

    for r in range(ROWS - INAROW + 1):
        for c in range(COLS - INAROW + 1):
            windows.append([(r + i, c + i) for i in range(INAROW)])

    for r in range(ROWS - INAROW + 1):
        for c in range(INAROW - 1, COLS):
            windows.append([(r + i, c - i) for i in range(INAROW)])

    return windows

def evaluate():
    total_score = 0

    for window in ALL_WINDOWS:
        cells = [board[r][c] for r,c in window]

        p_count = cells.count(player_id)
        o_count =  cells.count(opponent_id)
        empty = cells.count(0)

        if p_count == 4:
            total_score += 10000
        elif p_count == 3 and empty == 1:
            total_score += 50
        elif p_count == 2 and empty == 2:
            total_score += 10
        
        if o_count == 3 and empty == 1:
            total_score -= 80
        elif o_count == 4:
            total_score -= 10000
    
    return total_score

def dfs(depth):
    # print(board, file=f)
    # f.write("\n\n")
    is_player = depth % 2 == 0
    current_value = player_id if is_player else opponent_id
    valid_moves = [col for col in range(COLS)
                   if board[0][col] == 0]
    bst_val = -INF if is_player else INF
    bst_col = None
    if depth == MAX_DEPTH:
        return evaluate(), 0
    for j in valid_moves:
        i = 0
        while i + 1 < ROWS and board[i + 1][j] == 0:
            i += 1
        board[i][j] = current_value
        val, _ = dfs(depth + 1)
        if is_player:
            if val > bst_val:
                bst_val = val
                bst_col = j
        else:
            if val < bst_val:
                bst_val = val
                bst_col = j
        board[i][j] = 0
    # print(bst_col, file=f)
    return bst_val, bst_col

def pure_minimax(observation, configuration):
    global ALL_WINDOWS, ROWS, COLS, INAROW, board, player_id, opponent_id, f

    f = open("log.txt", "a")
    player_id = observation.mark
    opponent_id = 3 - player_id
    ROWS = configuration.rows
    COLS = configuration.columns
    board = np.array(observation.board).reshape(ROWS, COLS)
    INAROW = configuration.inarow
    ALL_WINDOWS = get_all_windows()

    # print("Player ID: ", player_id, file=f)

    _, col = dfs(0)
    f.close()
    return col

    