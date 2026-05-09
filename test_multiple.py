from kaggle_environments import make
from alpha_beta_bitsboard_minimax import alpha_beta_bitsboard_minimax
from alpha_beta_minimax import alpha_beta_minimax

env = make("connectx", debug=False)

trainer = env.train([None, alpha_beta_minimax])

wins = 0
losses = 0
draws = 0
total_games = 10

print(f"Starting {total_games} games...")

for i in range(total_games):
    obs = trainer.reset()
    done = False

    while not done:
        action = alpha_beta_bitsboard_minimax(obs, env.configuration)
        obs, reward, done, info = trainer.step(action)
    
    if reward == 1:
        wins += 1
    elif reward == -1:
        losses += 1
    else:
        draws += 1

    
    print(f"Game {i+1}/{total_games} complete.")

print("\n--- Final Results ---")
print(f"Wins: {wins} | Losses: {losses} | Draws: {draws}")
print(f"Win Rate: {(wins/total_games)*100}%")
