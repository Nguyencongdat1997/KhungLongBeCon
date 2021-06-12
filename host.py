from environment.env import Environment
from players.random_player import RandomPlayer
from players.d3q.agent import Agent

if __name__ == "__main__":
    env = Environment()
    player = Agent(lr=0.005, gamma=0.99, n_actions=env.n_action, epsilon=0.0, batch_size=1,
                 input_dims=env.observation_shape, epsilon_dec=0, epsilon_end=0.01,
                 mem_size=1, replace=1, coldstart=0)
    player.load_model(train_dir='./trained_models', learned_steps=1208)

    episodes = 3
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
