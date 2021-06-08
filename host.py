from environment.environment import Environment
from players.random_player import RandomPlayer

if __name__ == "__main__":
    player = RandomPlayer()
    env = Environment()

    episodes = 2
    steps_counter = 0
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            steps_counter += 1
            action = player.play(state)
            reward, next_state, done = env.step(action)
            score += reward
            state = next_state
        print('Episode:{} Steps:{} Score:{}'.format(episode, steps_counter, score))
    env.close()
