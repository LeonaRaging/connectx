import numpy as np
import time

MAX_TIME = 1.4 
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

def get_winning_mask(p, mask):
    """Returns a bitmask of empty squares that would complete a 4-in-a-row."""
    board_mask = 0x1FE3F7EFDFBF7F
    win = 0
    for s in [7, 1, 6, 8]:
        win |= (p & (p >> s) & (p >> 2*s)) << 3*s
        win |= (p & (p << s) & (p << 2*s)) >> 3*s
        win |= (p & (p >> s) & (p >> 3*s)) << 2*s
        win |= (p & (p << s) & (p << 3*s)) >> 2*s
        win |= (p & (p >> 2*s) & (p >> 3*s)) << s
        win |= (p & (p << 2*s) & (p << 3*s)) >> s
    return win & board_mask & ~mask

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
    is_agent = depth % 2 == 0
    
    if depth % 512 == 0:
        if time.time() - start_time > MAX_TIME:
            out_of_time = True
            return 0, None

    if check_win(opp_pos if is_agent else agent_pos):
        return -(WIN_SCORE - depth) if is_agent else (WIN_SCORE + depth), None

    mask = agent_pos | opp_pos
    my_pos = agent_pos if is_agent else opp_pos
    thr_pos = opp_pos if is_agent else agent_pos

    my_win_mask = get_winning_mask(my_pos, mask)
    playable_mask = 0
    for c in range(7):
        if heights[c] % 7 != 6:
            playable_mask |= (1 << heights[c])
    
    win_now = my_win_mask & playable_mask
    if win_now:
        winning_col = (win_now.bit_length() - 1) // 7
        return (WIN_SCORE - depth) if is_agent else -(WIN_SCORE - depth), winning_col

    opp_win_mask = get_winning_mask(thr_pos, mask)
    direct_threats = opp_win_mask & playable_mask
    if direct_threats.bit_count() > 1:
        return -(WIN_SCORE - (depth + 1)) if is_agent else (WIN_SCORE - (depth + 1)), None
    
    forced_move = None
    if direct_threats.bit_count() == 1:
        forced_move = (direct_threats.bit_length() - 1) // 7
    
    candidate_moves = [forced_move] if forced_move is not None else MOVE_ORDER
    valid_moves = []
    for col in candidate_moves:
        if heights[col] % 7 != 6:
            if not (opp_win_mask & (1 << (heights[col] + 1))):
                valid_moves.append(col)
    
    if not valid_moves:
        valid_moves = [c for c in MOVE_ORDER if heights[c] % 7 != 6]
        if not valid_moves: return 0, None

    move_priorities = []
    order_map = {col: i for i, col in enumerate(MOVE_ORDER)}
    
    for col in valid_moves:
        move = 1 << heights[col]
        score = get_winning_mask(my_pos | move, mask | move).bit_count()
        move_priorities.append((col, score, order_map[col]))
    
    move_priorities.sort(key=lambda x: (-x[1], x[2]))
    ordered_moves = [x[0] for x in move_priorities]

    board_hash = (agent_pos, opp_pos, is_agent)
    tt_entry = transposition_table.get(board_hash)
    best_prev_move = tt_entry[2] if tt_entry else None
    
    if best_prev_move is not None and best_prev_move in ordered_moves:
        ordered_moves.remove(best_prev_move)
        ordered_moves.insert(0, best_prev_move)

    if depth == limit:
        return evaluate_bitwise(agent_pos, opp_pos), None
    

    remaining_depth = limit - depth
    if tt_entry and tt_entry[1] >= remaining_depth:
        return tt_entry[0], best_prev_move
        
    bst_col = ordered_moves[0]
    bst_val = -INF if is_agent else INF

    for i, col in enumerate(ordered_moves):
        move = 1 << heights[col]
        if is_agent: agent_pos ^= move
        else: opp_pos ^= move
        heights[col] += 1
        
        if i == 0:
            val, _ = dfs(depth + 1, limit, alpha, beta, agent_pos, opp_pos, heights)
        else:
            if is_agent:
                val, _ = dfs(depth + 1, limit, alpha, alpha + 1, agent_pos, opp_pos, heights)
                if alpha < val < beta:
                    val, _ = dfs(depth + 1, limit, val, beta, agent_pos, opp_pos, heights)
            else:
                val, _ = dfs(depth + 1, limit, beta - 1, beta, agent_pos, opp_pos, heights)
                if alpha < val < beta:
                    val, _ = dfs(depth + 1, limit, alpha, val, agent_pos, opp_pos, heights)

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

def alpha_beta_bitsboard_minimax_iterative_deepening_null_window(observation, configuration):
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