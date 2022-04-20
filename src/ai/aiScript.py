from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython
from pythonEnvironment import pythonTetris

import tensorflow as tf
import matplotlib.pyplot as plt

from tempfile import TemporaryFile

from tensorflow import train

import numpy as np

import os
import tempfile

import reverb

from tf_agents.agents.dqn import dqn_agent

from tf_agents.agents.ddpg import critic_network
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.specs import tensor_spec
from tf_agents.specs import BoundedArraySpec
from tf_agents.networks import sequential
from tf_agents.networks import actor_distribution_network
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import py_epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network

import time

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units)

#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

num_iterations = 1000000 # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 20000000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}

epsilon = 0.2

# initial_collect_steps = 10000 # @param {type:"integer"}
# collect_steps_per_iteration = 1 # @param {type:"integer"}
# replay_buffer_max_length = 10000 # @param {type:"integer"}

# batch_size = 256 # @param {type:"integer"}

# critic_learning_rate = 3e-4 # @param {type:"number"}
# actor_learning_rate = 3e-4 # @param {type:"number"}
# alpha_learning_rate = 3e-4 # @param {type:"number"}
# target_update_tau = 0.005 # @param {type:"number"}
# target_update_period = 1 # @param {type:"number"}
# gamma = 0.99 # @param {type:"number"}
# reward_scale_factor = 1.0 # @param {type:"number"}

# actor_fc_layer_params = (256, 256)
# critic_joint_fc_layer_params = (256, 256)

# log_interval = 5000 # @param {type:"integer"}

# num_eval_episodes = 20 # @param {type:"integer"}
# eval_interval = 10000 # @param {type:"integer"}

# policy_save_interval = 5000 # @param {type:"integer"}

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

'''
Creates environments
'''
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


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])
global_step = tf.compat.v1.train.get_or_create_global_step()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

# strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

# observation_spec, action_spec, time_step_spec = (
#       spec_utils.get_tensor_specs(train_env))

# with strategy.scope():
#   critic_net = critic_network.CriticNetwork(
#         (observation_spec, action_spec),
#         observation_fc_layer_params=None,
#         action_fc_layer_params=None,
#         joint_fc_layer_params=critic_joint_fc_layer_params,
#         kernel_initializer='glorot_uniform',
#         last_kernel_initializer='glorot_uniform')
  
#   actor_net = actor_distribution_network.ActorDistributionNetwork(
#       observation_spec,
#       action_spec,
#       fc_layer_params=actor_fc_layer_params,
#       continuous_projection_net=(
#           tanh_normal_projection_network.TanhNormalProjectionNetwork))
  
#   train_step = train_utils.create_train_step()

#   agent = sac_agent.SacAgent(
#         time_step_spec,
#         action_spec,
#         actor_network=actor_net,
#         critic_network=critic_net,
#         actor_optimizer=tf.keras.optimizers.Adam(
#             learning_rate=actor_learning_rate),
#         critic_optimizer=tf.keras.optimizers.Adam(
#             learning_rate=critic_learning_rate),
#         alpha_optimizer=tf.keras.optimizers.Adam(
#             learning_rate=alpha_learning_rate),
#         target_update_tau=target_update_tau,
#         target_update_period=target_update_period,
#         td_errors_loss_fn=tf.math.squared_difference,
#         gamma=gamma,
#         reward_scale_factor=reward_scale_factor,
#         train_step_counter=train_step)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
print("script checkpoint 1")
table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

py_driver.PyDriver(
    python_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=False),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

print("script checkpoint 2")

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

print("script checkpoint 3")

iterator = iter(dataset)
# print(iterator)
# print(iterator.next())

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

print("script checkpoint 4")

# Reset the train step.
agent.train_step_counter.assign(0)


print("between 1")
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print("between 2")

return_file = '/home/block5/Documents/GitHub/tetris/src/ai/checkpoint/returns.npy'
if os.path.exists(return_file):
  for i in range(10):
    print("LOADING")
  returns =  np.load(return_file).tolist()
  returns.append(avg_return)
else:
  returns = [avg_return]

print("between 3")
# Reset the environment.
time_step = train_py_env.reset()

print("script checkpoint 5")

# Create a driver to collect experience.
# collect_driver = py_driver.PyDriver(
#     python_env,
#     py_tf_eager_policy.PyTFEagerPolicy(
#       agent.collect_policy, use_tf_function=True),
#     [rb_observer],
#     max_steps=collect_steps_per_iteration)

collect_driver = py_driver.PyDriver(
    python_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

checkpoint_dir = '/home/block5/Documents/GitHub/tetris/src/ai/checkpoint'
global_step = tf.compat.v1.train.get_or_create_global_step()
train_checkpointer = common.Checkpointer(
  ckpt_dir=checkpoint_dir,
  max_to_keep=1,
  agent=agent,
  policy=agent.policy,
  replay_buffer=replay_buffer,
  global_step=global_step
)
train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

print("script checkpoint 6")
# try catch allows user to stop training and still see graph
try:  
  for _ in range(num_iterations):

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
      print()
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)
finally:
  
  train_checkpointer.save(global_step)
  np.save(return_file, returns)
  
  # iterations = range(0, num_iterations + 1, eval_interval)
  iterations = range(0, len(returns))
  plt.plot(iterations, returns)
  print("it: {}".format(iterations))
  print("plotted")
  plt.ylabel('Average Return')
  plt.xlabel('Iterations')
  # plt.ylim(top=250)
  plt.show()



# utils.validate_py_environment(python_env)


# print('action_spec:', tf_env.action_spec())
# print('time_step_spec.observation:', tf_env.time_step_spec().observation)
# print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
# print('time_step_spec.discount:', tf_env.time_step_spec().discount)
# print('time_step_spec.reward:', tf_env.time_step_spec().reward)

# action = np.array((2,), dtype=np.int32)


# rewards = []
# steps = []
# number_of_episodes = 2


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
