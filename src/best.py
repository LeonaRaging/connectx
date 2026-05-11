import numpy as np

WIDTH = 7
HEIGHT = 6

moves = 0
current_position = 0
mask = 0
column_order = [0] * WIDTH
transposition_table = {}

def move_score(move):
    return compute_winning_position(current_position | move, mask).bit_count()

for i in range(WIDTH):
    column_order[i] = WIDTH // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2

def bottom(width, height):
    return 0 if width == 0 else bottom(width - 1, height) | 1 << (width - 1) * (height + 1)

def top_mask_col(col):
    return 1 << ((HEIGHT - 1) + col * (HEIGHT + 1))

def bottom_mask_col(col):
    return 1 << col * (HEIGHT + 1)

def column_mask(col):
    return ((1 << HEIGHT) - 1) << col * (HEIGHT + 1)

def preprocessing(observation):
    global current_position, mask, moves

    current_position = 0
    mask = 0
    moves = 0

    grid = np.array(observation.board).reshape(HEIGHT, WIDTH)

    moves = np.count_nonzero(grid)

    current_player = 1 if moves % 2 == 0 else 2

    for c in range(WIDTH):
        for r in range(HEIGHT):

            piece = grid[HEIGHT - 1 - r][c]

            if piece != 0:

                bit = 1 << (c * (HEIGHT + 1) + r)

                mask |= bit

                if piece == current_player:
                    current_position |= bit

def play(col):
    global moves, current_position, mask

    current_position ^= mask
    mask |= mask + bottom_mask_col(col)
    moves += 1


def un_play(previous_position, previous_mask):
    global moves, current_position, mask

    moves -= 1
    current_position = previous_position
    mask = previous_mask


def can_win_next():
    return winning_position() & possible() != 0

def possible_non_loosing_moves():
    possible_mask = possible()
    opponent_win = opponent_winning_position()
    forced_moves = possible_mask & opponent_win
    
    if forced_moves:
        if forced_moves & (forced_moves - 1):
            return 0
        else:
            possible_mask = forced_moves
    
    return possible_mask & ~(opponent_win >> 1)

def can_play(col):
    return (mask & top_mask_col(col)) == 0

def is_winning_move(col):
    return winning_position() & possible() & column_mask(col) != 0

def winning_position():
    return compute_winning_position(current_position, mask)

def opponent_winning_position():
    return compute_winning_position(current_position ^ mask, mask)

def possible():
    return (mask + bottom_mask) & board_mask

def compute_winning_position(position, mask):
    r = (position << 1) & (position << 2) & (position << 3)

    p = (position << (HEIGHT + 1)) & (position << (2 * (HEIGHT + 1)))

    r |= p & (position << (3 * (HEIGHT + 1)))
    r |= p & (position >> (HEIGHT + 1))

    p >>= 3 * (HEIGHT + 1)

    r |= p & (position << (HEIGHT + 1))
    r |= p & (position >> (3 * (HEIGHT + 1)))

    p = (position << HEIGHT) & (position << (2 * HEIGHT))

    r |= p & (position << (3 * HEIGHT))
    r |= p & (position >> HEIGHT)

    p >>= 3 * HEIGHT

    r |= p & (position << HEIGHT)
    r |= p & (position >> (3 * HEIGHT))

    p = (position << (HEIGHT + 2)) & (position << (2 * (HEIGHT + 2)))

    r |= p & (position << (3 * (HEIGHT + 2)))
    r |= p & (position >> (HEIGHT + 2))

    p >>= 3 * (HEIGHT + 2)

    r |= p & (position << (HEIGHT + 2))
    r |= p & (position >> (3 * (HEIGHT + 2)))

    return r & (board_mask ^ mask)

bottom_mask = bottom(WIDTH, HEIGHT)
board_mask = bottom_mask * ((1 << HEIGHT) - 1)

def key():
    return current_position + mask


def negamax(alpha, beta):
    global moves

    next = possible_non_loosing_moves()

    if next == 0:
        return -(WIDTH * HEIGHT - moves) // 2
    
    if moves >= WIDTH * HEIGHT - 2:
        return 0

    min = -(WIDTH * HEIGHT - 2 - moves) // 2

    if alpha < min:
        alpha = min
        if alpha >= beta:
            return alpha

    max = (WIDTH * HEIGHT - 1 - moves) // 2

    position_key = key()

    if position_key in transposition_table:
        bound_type, val = transposition_table[position_key]

        if bound_type == "LOWER":
            if alpha < val:
                alpha = val
        else:
            if beta > val:
                beta = val
    
    if alpha >= beta:
        return alpha

    if beta > max:
        beta = max
        if alpha >= beta:
            return beta
        
    moves_list = []

    for x in range(WIDTH):
        col = column_order[x]

        move = next & column_mask(col)

        if move:
            moves_list.append((move_score(move), col))
    
    moves_list.sort(reverse=True)

    for _, col in moves_list:
        prev_position = current_position
        prev_mask = mask

        play(col)

        score = -negamax(-beta, -alpha)

        un_play(prev_position, prev_mask)

        if score >= beta:
            transposition_table[position_key] = ("LOWER", score)
            return score

        if score > alpha:
            alpha = score
    transposition_table[position_key] = ("UPPER", alpha)
    return alpha

def solve():
    if can_win_next():
        return (WIDTH * HEIGHT + 1 - moves) // 2

    min = -(WIDTH * HEIGHT - moves) // 2
    max = (WIDTH * HEIGHT + 1 - moves) // 2

    while min < max:
        med = min + (max - min) // 2
        if med <= 0 and min // 2 < med:
            med = min // 2
        elif med >= 0 and max // 2 > med:
            med = max // 2
        r = negamax(med, med + 1)
        if r <= med:
            max = r
        else:
            min = r
    return min

def best(observation, configuration):
    global transposition_table

    transposition_table.clear()

    preprocessing(observation)

    if can_win_next():
        for c in range(WIDTH):
            if can_play(c) and is_winning_move(c):
                return c

    best_col = -1
    best_score = -100

    for x in range(WIDTH):
        col = column_order[x]

        if can_play(col):
            prev_position = current_position
            prev_mask = mask

            play(col)

            score = -solve()

            un_play(prev_position, prev_mask)

            if score > best_score:
                best_score = score
                best_col = col

    return best_col