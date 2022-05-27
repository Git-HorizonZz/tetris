import os
print(os.path.dirname(os.path.realpath(__file__)))

from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython
from pythonEnvironment import pythonTetris

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

import reverb

from tf_agents.agents.dqn import dqn_agent

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.specs import tensor_spec
from tf_agents.specs import BoundedArraySpec
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

import time

num_iterations = 1000000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}

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

for i in range(40):
    javaTalker.go_to_location(np.int32(i))
    print(i)
    while not javaTalker.just_collided():
        pass