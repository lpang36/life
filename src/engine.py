import numpy as np
import scipy

from bond import Bond
from display import Display
from gradient import Gradient
from grid import Grid
from motion import Motion
from repopulate import Repopulate


class Engine:
    def __init__(self, params):
        self._params = params
        repopulate = Repopulate(params)
        # x, y, type
        # TODO: orientation, velocity, acceleration, temperature
        self._position_state, self._type_state = repopulate.random_initial_state()
        self._bond_state = scipy.sparse.lil_matrix(
            (params.particle.num_initial_particles, params.particle.num_initial_particles))
        dims = np.array([params.grid.max_x, params.grid.max_y])
        self._attractor_gradient_state = np.random.rand(
            params.gradient.num_attractors, 2) * dims
        self._repeller_gradient_state = np.random.rand(
            params.gradient.num_repellers, 2) * dims
        self.transforms = [cls(params)
                           for cls in (Grid, Gradient, Bond, Motion)] + [repopulate]
        self.sinks = [cls(params) for cls in (Display,)]

    def step(self):
        states = [
            'position_state',
            'type_state',
            'bond_state',
            'attractor_gradient_state',
            'repeller_gradient_state',
        ]
        results = {state: getattr(self, f'_{state}') for state in states}
        for transform in self.transforms:
            results = {**results, **transform(**results)}
        for state in states:
            setattr(self, f'_{state}', results[state])
        for sink in self.sinks:
            sink(**results)
