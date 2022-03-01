from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython
from pythonEnvironment import pythonTetris

import tensorflow as tf

import numpy as np

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

import time

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

# tf_env = tf_py_environment.TFPyEnvironment(python_env)


# print('action_spec:', tf_env.action_spec())
# print('time_step_spec.observation:', tf_env.time_step_spec().observation)
# print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
# print('time_step_spec.discount:', tf_env.time_step_spec().discount)
# print('time_step_spec.reward:', tf_env.time_step_spec().reward)

action = np.array((8,4), dtype=np.int32)

# # print(time_step)
for _ in range(5):
    time_step = python_env._reset()
    while not time_step.is_last():
    # # time.sleep(1)
        terminal.println("new step")
        time_step = python_env.step(action)
    #   print(time_step)

# rewards = []
# steps = []
# number_of_episodes = 2

# for _ in range(number_of_episodes):
#     reward_t = 0
#     steps_t = 0
#     tf_env.reset()
#     while True:
#         action = tf.random.uniform([10,4], 0, 10, dtype=tf.int32)
#         next_time_step = tf_env.step(action)
#         if tf_env.current_time_step().is_last():
#             break
#         steps_t += 1
#         reward_t += next_time_step.reward.numpy()
#     rewards.append(reward_t)
#     steps.append(steps_t)

# tf_env.close()

# while True:
#     print(javaTalker.get_episode_over())
#     if javaTalker.get_episode_over():
#         javaTalker.restart()
#     time.sleep(1)