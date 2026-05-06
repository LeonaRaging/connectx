def my_agent(observation, configuration):
    import random
    # Get list of valid moves (columns with empty spaces)
    valid_moves = [col for col in range(configuration.columns) 
                   if observation.board[col] == 0]
    # Return a random valid move
    return random.choice(valid_moves)