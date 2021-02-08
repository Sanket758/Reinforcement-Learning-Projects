"""
    Author: Sanket S. Gadge
    Description: This script uses OpenAI GYM to create an Environment, using Keras-RL library we will create an agent
     which will learn to balance the cart pole on its own.
    Reference: https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/
                https://keras.io/examples/rl/actor_critic_cartpole/
"""
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam

from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents import DQNAgent

import warnings
warnings.filterwarnings('ignore')

import gym  # will be needed to create our environment

# creating a cartpole environment check: https://gym.openai.com/docs/ for other envs.
env = gym.make('CartPole-v0')

# variables to hold state_size and number of actions
STATE_SIZE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n
ENV_NAME = 'CartPole-v0'

env.reset() # reset the env


# Building a neural network which will take a vector of states and output the best action to take
def build_model(state_size, number_of_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    # x = Dense(16, activation='relu')(x)
    output = Dense(number_of_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model


# creating model object
model = build_model(STATE_SIZE, NUM_ACTIONS)

# Creating the memory for our network in order to implement experience replay
memory = SequentialMemory(limit=50000, window_length=1)

# We will use the Greedy policy here and LinearAnneledPolicy will handle the decay
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

# Creating a DQNAgent
# Read more about agents: https://keras-rl.readthedocs.io/en/latest/agents/overview/
dqn = DQNAgent(model=model,
               nb_actions=NUM_ACTIONS,
               memory=memory,
               nb_steps_warmup=10,
               target_model_update=1e-2,
               policy=policy
            )

# Compile the model
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae', 'mse'])


# Defining callbacks....will save the model and logfile
def build_callbacks(env_name):
    checkpoint_weights_filename = f'dqn_{env_name}_weights.h5f'
    log_filename = f'dqn_{env_name}_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                       interval=5000)]

    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


callbacks = build_callbacks(ENV_NAME)
# Fit the model
dqn.fit(env, nb_steps=10000, visualize=False,
        verbose=2, callbacks=callbacks)

# Run test
dqn.test(env, nb_episodes=15, visualize=True)
