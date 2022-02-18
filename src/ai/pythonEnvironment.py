print("hello from python!")
import tensorflow as tf
print("tf import finished")

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from javaToPython import JavaToPython

import numpy as np

class pythonTetris(py_environment.PyEnvironment):
    def __init__(self):
        self.java_talker = JavaToPython()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(10, 4), dtype=np.int32, minimum=0, maximum=40, name='play')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12,23), dtype=np.int32, minimum=0, maximum=1, name='board')
        self._state = np.zeros(shape=(12,23))
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros(shape=(12,23))
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if self.java_talker.get_episode_over():
            self.java_talker.restart()
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.int32), -1)
        else:
            self._state = self.java_talker.get_python_wall()

            

        # TODO: analyze wall
