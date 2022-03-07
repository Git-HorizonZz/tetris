from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython
from pythonEnvironment import pythonTetris

import tensorflow as tf

import numpy as np

from tf_agents.agents.dqn import dqn_agent

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

from tf_agents.specs import tensor_spec
from tf_agents.specs import BoundedArraySpec

from tf_agents.networks import sequential

from tf_agents.utils import common

import time


num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


'''
Connects to java script
'''
gateway = JavaGateway()
tetris_game = gateway.jvm.tetris.TetrisDriver()
actions_obj = tetris_game.getActionsObject()
tetris_UI = tetris_game.getGameUI()
terminal = gateway.jvm.System.out

javaTalker = JavaToPython(gateway)

terminal.println("hello from python")

print("python")
python_env = pythonTetris(javaTalker)
python_env.reset()

tf_env = tf_py_environment.TFPyEnvironment(python_env)

train_py_env = pythonTetris(javaTalker)
eval_py_env = pythonTetris(javaTalker)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(python_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
print(num_actions)
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

# agent.initialize()

# utils.validate_py_environment(python_env)


# print('action_spec:', tf_env.action_spec())
# print('time_step_spec.observation:', tf_env.time_step_spec().observation)
# print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
# print('time_step_spec.discount:', tf_env.time_step_spec().discount)
# print('time_step_spec.reward:', tf_env.time_step_spec().reward)

action = np.array((2,), dtype=np.int32)


rewards = []
steps = []
number_of_episodes = 2


# for _ in range(number_of_episodes):
#     reward_t = 0
#     steps_t = 0
#     while not tf_env.current_time_step().is_last():
#         # action_test = action_test.from_array(np.array([np.random.randint(0,10), np.random.randint(0,4)], np.int32), name='play')

#         print(action_test)

#         action = [np.random.randint(0,10), np.random.randint(0,4)]
#         next_time_step = tf_env.step(action)
#         # print(tf_env.time_step_spec().reward)
#         steps_t += 1
#         reward_t += next_time_step.reward#.numpy()
#     rewards.append(reward_t)
#     steps.append(steps_t)
#     tf_env.reset()

tf_env.close()
train_env.close()
eval_env.close()