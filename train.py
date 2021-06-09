from environment.environment import Environment
from players.d3q.agent import Agent

if __name__ == "__main__":
    env = Environment()
    d3qn = Agent(lr=0.005, gamma=0.99, n_actions=env.n_action, epsilon=1.0, batch_size=2,
                 input_dims=env.observation_shape, replace=2)

    n_games = 1
    d3qn.train(env, n_games)

    train_dir = './trained_models'
    d3qn.epsilon = 0.0
    d3qn.save_model(train_dir)