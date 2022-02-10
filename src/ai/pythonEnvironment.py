import tensorflow as tf
print("tf import finished")

from tf_agents.environments import py_environment

import numpy

class pythonTetris(py_environment.PyEnvironment):
    def __init__(self):
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