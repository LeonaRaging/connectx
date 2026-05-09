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
                    row_str += " 1 |"
                elif (opp_pos >> bit_index) & 1:
                    row_str += " 2 |"
                else:
                    row_str += "   |"
            f.write(row_str + "\n")
        f.write("-----------------------------")
        f.write("\n")
        f.write(str(evaluate_bitwise(agent_pos, opp_pos)))
        f.write("\n")

def check_win(pos):
    for shift in [7, 6, 8, 1]:
        m = pos & (pos >> shift)
        if m & (m >> (2 * shift)): return True
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
    score = 0
    
    score += (agent_pos & CENTER_MASK).bit_count() * 3
    score -= (opp_pos & CENTER_MASK).bit_count() * 3
    
    for shift in [1, 7, 8, 6]:
        p_pairs = agent_pos & (agent_pos >> shift)
        score += p_pairs.bit_count()                
        score += (p_pairs & (p_pairs >> shift)).bit_count() * 5 
        
        o_pairs = opp_pos & (opp_pos >> shift)
        score -= o_pairs.bit_count()
        score -= (o_pairs & (o_pairs >> shift)).bit_count() * 5
        
    return score

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
    

    