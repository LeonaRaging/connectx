import numpy as np
import time

MAX_TIME = 1.5
ABSOLUTE_MAX_DEPTH = 20
WIN_SCORE = 1000000
INF = float('inf')

CENTER_MASK = 0x3F << 21 
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]
transposition_table = {}
start_time = 0
out_of_time = False

def check_win(pos):
    for shift in [7, 6, 8, 1]:
        m = pos & (pos >> shift)
        if m & (m >> (2 * shift)): return True
    return False

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

def dfs(depth, limit, alpha, beta, agent_pos, opp_pos, heights):
    global out_of_time
    
    if depth % 2 == 0:
        if time.time() - start_time > MAX_TIME:
            out_of_time = True
            return 0, None

    is_agent = depth % 2 == 0

    if check_win(opp_pos if is_agent else agent_pos):
        return -(WIN_SCORE - depth) if is_agent else (WIN_SCORE + depth), None

    board_hash = (agent_pos, opp_pos, is_agent)
    tt_entry = transposition_table.get(board_hash)
    
    best_prev_move = tt_entry[2] if tt_entry else None
    
    current_move_order = MOVE_ORDER
    if best_prev_move is not None:
        current_move_order = [best_prev_move] + [m for m in MOVE_ORDER if m != best_prev_move]

    valid_moves = [c for c in current_move_order if heights[c] % 7 != 6]

    if depth == limit or not valid_moves:
        return evaluate_bitwise(agent_pos, opp_pos), None
    
    remaining_depth = limit - depth
    if tt_entry:
        cached_val, cached_rem_depth, _ = tt_entry
        if cached_rem_depth >= remaining_depth:
            return cached_val, best_prev_move
        
    bst_col = valid_moves[0]
    bst_val = -INF if is_agent else INF

    for col in valid_moves:
        move = 1 << heights[col]
        if is_agent: agent_pos ^= move
        else: opp_pos ^= move
        heights[col] += 1
        
        val, _ = dfs(depth + 1, limit, alpha, beta, agent_pos, opp_pos, heights)

        heights[col] -= 1
        if is_agent: agent_pos ^= move
        else: opp_pos ^= move

        if out_of_time: return 0, None

        if is_agent:
            if val > bst_val: bst_val, bst_col = val, col
            alpha = max(alpha, bst_val)
        else:
            if val < bst_val: bst_val, bst_col = val, col
            beta = min(beta, bst_val)
        
        if beta <= alpha: break

    transposition_table[board_hash] = (bst_val, remaining_depth, bst_col) 
    return bst_val, bst_col

def alpha_beta_bitsboard_minimax_iterative_deepening(observation, configuration):
    global start_time, out_of_time
    start_time = time.time()
    out_of_time = False

    agent_pos, opp_pos = 0, 0
    heights = [c * 7 for c in range(7)]
    agent_id = observation.mark

    for r in range(5, -1, -1):
        for c in range(7):
            piece = observation.board[r * 7 + c]
            if piece != 0:
                move = 1 << heights[c]
                if piece == agent_id: agent_pos ^= move
                else: opp_pos ^= move
                heights[c] += 1
    
    best_col_overall = [c for c in MOVE_ORDER if heights[c] % 7 != 6][0]

    for current_limit in range(1, ABSOLUTE_MAX_DEPTH + 1):
        val, col = dfs(0, current_limit, -INF, INF, agent_pos, opp_pos, heights)
        
        if not out_of_time:
            best_col_overall = col
        else:
            break
        
        if val >= WIN_SCORE - 100 or val <= -WIN_SCORE + 100:
            break

    return int(best_col_overall)