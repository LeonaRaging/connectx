from kaggle_environments import make
from src.alpha_beta_minimax import alpha_beta_minimax
from src.alpha_beta_bitsboard_minimax import alpha_beta_bitsboard_minimax

# Initialize the environment
env = make("connectx", debug=True)

# Run a game
# Your agent is in the first position, "random" is in the second
env.run([alpha_beta_bitsboard_minimax, alpha_beta_minimax])

with open("game.html", "w") as f:
    f.write(env.render(mode="html"))

# Print the result
print("Game finished!")
print(f"Winner: {env.state[0].reward}") # None if tie, 1 if P1 wins, 2 if P2 wins