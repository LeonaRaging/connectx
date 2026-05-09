import numpy as np

MAX_DEPTH = 11
WIN_SCORE = 1000000
INF = float('inf')

CENTER_MASK = 0x3f << 21
BOARD_MASK = 0
for c in range(7):
    BOARD_MASK |= (0x3f << (c * 7))

MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

transposition_table = {}

def print_board(agent_pos, opp_pos):

    with open("log.txt", "a") as f:
        f.write("\n  0   1   2   3   4   5   6 \n")
        f.write("-----------------------------\n")
        for row in range(5, -1, -1):
            row_str = "|"
            for col in range(7):
                bit_index = col * 7 + row
                if (agent_pos >> bit_index) & 1:
                    row_str += " X |"
                elif (opp_pos >> bit_index) & 1:
                    row_str += " 0 |"
                else:
                    row_str += "   |"
            f.write(row_str + "\n")
        f.write("-----------------------------")
        f.write("\n")
        f.write(str(evaluate_bitwise(agent_pos, opp_pos)))
        f.write("\n")

def check_win(pos):
    # Horizontal
    m = pos & (pos >> 7)
    if m & (m >> 14): return True
    # Diagonal 1
    m = pos & (pos >> 6)
    if m & (m >> 12): return True
    # Diagonal 2
    m = pos & (pos >> 8)
    if m & (m >> 16): return True
    # Vertical
    m = pos & (pos >> 1)
    if m & (m >> 2): return True
    return False

def count_possible_4s(pos_mask):
    m = pos_mask & (pos_mask >> 7)
    count = (m & (m >> 14)).bit_count()
    
    m = pos_mask & (pos_mask >> 1)
    count += (m & (m >> 2)).bit_count()
    
    m = pos_mask & (pos_mask >> 6)
    count += (m & (m >> 12)).bit_count()
    
    m = pos_mask & (pos_mask >> 8)
    count += (m & (m >> 16)).bit_count()
    
    return count

def evaluate_bitwise(agent_pos, opp_pos):
    agent_possible = count_possible_4s(BOARD_MASK & ~opp_pos)
    
    opp_possible = count_possible_4s(BOARD_MASK & ~agent_pos)
    
    return agent_possible - opp_possible

def dfs(depth, alpha, beta, agent_pos, opp_pos, heights):
    # print_board(agent_pos, opp_pos)

    is_agent = depth % 2 == 0

    if check_win(opp_pos if is_agent else agent_pos):
        return -(WIN_SCORE - depth) if is_agent else (WIN_SCORE + depth), None

    valid_moves = [c for c in MOVE_ORDER if heights[c] % 7 != 6]

    if depth == MAX_DEPTH or not valid_moves:
        return evaluate_bitwise(agent_pos, opp_pos), None
    
    board_hash = (agent_pos, opp_pos, is_agent)
    remaining_depth = MAX_DEPTH - depth
    if board_hash in transposition_table:
        cached_val, cached_rem_depth = transposition_table[board_hash]
        if cached_rem_depth >= remaining_depth:
            return cached_val, None
        
    bst_col = valid_moves[0]

    if is_agent:
        bst_val = -INF
        for col in valid_moves:
            move = 1 << heights[col]
            agent_pos ^= move
            heights[col] += 1
            
            val, _ = dfs(depth + 1, alpha, beta, agent_pos, opp_pos, heights)

            heights[col] -= 1
            agent_pos ^= move

            if val > bst_val:
                bst_val = val
                bst_col = col
            
            alpha = max(alpha, bst_val)
            if beta <= alpha:
                break
    else:
        bst_val = INF
        for col in valid_moves:
            move = 1 << heights[col]
            opp_pos ^= move
            heights[col] += 1
            
            val, _ = dfs(depth + 1, alpha, beta, agent_pos, opp_pos, heights)
            
            # Undo move
            heights[col] -= 1
            opp_pos ^= move
            
            if val < bst_val:
                bst_val = val
                bst_col = col
            
            beta = min(beta, bst_val)
            if beta <= alpha:
                break
    transposition_table[board_hash] = (bst_val, remaining_depth) 
    return bst_val, bst_col

def alpha_beta_bitsboard_minimax(observation, configuration):
    agent_pos = 0
    opp_pos = 0

    heights = [c * 7 for c in range(7)]

    agent_id = observation.mark

    for r in range(5, -1, -1):
        for c in range(7):
            idx = r * 7 + c
            piece = observation.board[idx]
            if piece != 0:
                move = 1 << heights[c]
                if piece == agent_id:
                    agent_pos ^= move
                else:
                    opp_pos ^= move
                heights[c] += 1
    
    _, col = dfs(0, -INF, INF, agent_pos, opp_pos, heights)

    return col
    

    