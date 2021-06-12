from environment.env import Environment
from players.d3q.agent import Agent

if __name__ == "__main__":
    env = Environment()
    d3qn = Agent(lr=0.05, gamma=0.99, n_actions=env.n_action, epsilon=1.0, batch_size=2,
                 input_dims=env.observation_shape, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=256, replace=2, coldstart=256)
    #print(d3qn.q_active.summary())
    #print(d3qn.q_frozen.summary())
    n_games = 500
    d3qn.train(env, n_games)

    train_dir = './trained_models'
    d3qn.epsilon = 0.0
    d3qn.save_model(train_dir)