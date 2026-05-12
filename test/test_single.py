from kaggle_environments import make
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from alpha_beta_bitsboard_minimax_iterative_deepening_null_window import alpha_beta_bitsboard_minimax_iterative_deepening_null_window
from main import main

# Initialize the environment
env = make("connectx", debug=True)

# Run a game
# Your agent is in the first position, "random" is in the second
env.run([main, alpha_beta_bitsboard_minimax_iterative_deepening_null_window])

with open("game.html", "w") as f:
    f.write(env.render(mode="html"))

# Print the result
print("Game finished!")
print(f"Winner: {env.state[0].reward}") # None if tie, 1 if P1 wins, 2 if P2 wins