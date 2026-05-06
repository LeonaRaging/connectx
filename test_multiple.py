from kaggle_environments import make
from pure_minimax import pure_minimax

env = make("connectx", debug=False)

trainer = env.train([None, "random"])

wins = 0
losses = 0
draws = 0
total_games = 100

print(f"Starting {total_games} games...")

for i in range(total_games):
    obs = trainer.reset()
    done = False

    while not done:
        action = pure_minimax(obs, env.configuration)
        obs, reward, done, info = trainer.step(action)
    
    if reward == 1:
        wins += 1
    elif reward == -1:
        losses += 1
    else:
        draws += 1

    if (i + 1) % 10 == 0:
        print(f"Game {i+1}/{total_games} complete.")

print("\n--- Final Results ---")
print(f"Wins: {wins} | Losses: {losses} | Draws: {draws}")
print(f"Win Rate: {(wins/total_games)*100}%")
