print("hello from python!")
import tensorflow as tf
print("tf import finished")

from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from javaToPython import JavaToPython

import numpy as np

from waiting import wait

class pythonTetris(py_environment.PyEnvironment):
    def __init__(self, java_talker):
        self.java_talker = java_talker
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=39, name='play')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(288,), dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._state = np.zeros(shape=(288,))
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.java_talker.restart()
        self._state = np.zeros(shape=(288,))
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
            
        # only run step logic right after the piece lands
        # Restarts if episode is over
        if self._episode_ended:
            return self.reset()

        if self.java_talker.get_episode_over():
            # If episode is over, let environment know and give punishment of -2
            self.java_talker.restart()
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), -2)
        else:
            # print("step!")
            # Otherwise decide action and see wall
            # print(action)
            self.java_talker.go_to_location(action)
            self._state = self.java_talker.get_python_wall()
            

            # Waits to give reward until newest block collides
            # wait(lambda: self.java_talker.just_collided(), sleep_seconds=0.05)
            while not self.java_talker.just_collided():
                pass
            
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=self.java_talker.get_reward(),
                discount=0.9)
