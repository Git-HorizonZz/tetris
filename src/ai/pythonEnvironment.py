print("hello from python!")
import tensorflow as tf
print("tf import finished")

from tf_agents.environments import py_environment

from javaToPython import JavaToPython

import numpy

class pythonTetris(py_environment.PyEnvironment):
    def __init__(self):
        self.java_talker = JavaToPython()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(10, 4), dtype=np.int32, minimum=0, maximum=40, name='play')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(10,21), dtype=np.int32, minimum=0, maximum=1, name='board')
        self._state = numpy.zeros(shape=(10,21))
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = numpy.zeros(shape=(10,21))
        self._episode_ended = False
        return ts.restart(numpy.array([self._state], dtype=numpy.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # TODO: analyze wall
        # TODO: also, figure out how to get the wall in here
