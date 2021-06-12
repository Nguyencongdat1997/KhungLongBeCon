from environment.env import Environment
from players.d3q.agent import Agent

if __name__ == "__main__":
    env = Environment()
    d3qn = Agent(lr=0.005, gamma=0.99, n_actions=env.n_action, epsilon=0.9, batch_size=32,
                 input_dims=env.observation_shape, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=256, replace=4, coldstart=256)
    #print(d3qn.q_active.summary())
    #print(d3qn.q_frozen.summary())

    d3qn.train(env, n_games=150, n_save=50, train_dir='./trained_models')    
    