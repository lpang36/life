import numpy as np


class Gradient:
    def __init__(self, params):
        self._pole_motion = params.gradient.pole_motion
        self._max_x = params.grid.max_x
        self._max_y = params.grid.max_y

    def __call__(self, attractor_gradient_state, repeller_gradient_state, **kwargs):
        return {
            "attractor_gradient_state": self._move(attractor_gradient_state),
            "repeller_gradient_state": self._move(repeller_gradient_state)
        }

    def _move(self, state):
        state += np.random.rand(*state.shape) * 2 * \
            self._pole_motion - self._pole_motion
        return state.clip([0, 0], [self._max_x, self._max_y])
