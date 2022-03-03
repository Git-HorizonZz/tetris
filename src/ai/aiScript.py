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
python_env.reset()

# tf_env = tf_py_environment.TFPyEnvironment(python_env)


# print('action_spec:', tf_env.action_spec())
# print('time_step_spec.observation:', tf_env.time_step_spec().observation)
# print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
# print('time_step_spec.discount:', tf_env.time_step_spec().discount)
# print('time_step_spec.reward:', tf_env.time_step_spec().reward)

action = np.array((8,4), dtype=np.int32)

rewards = []
steps = []
number_of_episodes = 2


for _ in range(number_of_episodes):
    reward_t = 0
    steps_t = 0
    while not python_env.current_time_step().is_last():
        action = [np.random.randint(0,10), np.random.randint(0,4)]
        next_time_step = python_env.step(action)
        print("step done")
        steps_t += 1
        reward_t += next_time_step.reward#.numpy()
    rewards.append(reward_t)
    steps.append(steps_t)
    python_env.reset()

# tf_env.close()