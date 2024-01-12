import numpy as np
import scipy


class Repopulate:
    def __init__(self, params):
        self._max_x = params.grid.max_x
        self._max_y = params.grid.max_y
        self._num_initial_particles = params.particle.num_initial_particles
        self._initial_distribution = np.array(
            params.particle.initial_distribution)
        self._num_particle_types = params.particle.num_particle_types

    def random_initial_state(self):
        position_state = np.random.rand(self._num_initial_particles, 2)
        position_state *= np.array([self._max_x, self._max_y])
        return (
            position_state,
            np.random.choice(self._num_particle_types,
                             self._num_initial_particles, p=self._initial_distribution),
        )

    def __call__(self, position_state, type_state, bond_state, **kwargs):
        in_bounds = ((position_state[:, 0] >= 0) & (position_state[:, 1] >= 0) & (
            position_state[:, 0] <= self._max_x) & (position_state[:, 1] <= self._max_y))
        random_pos, random_type = self.random_initial_state()
        # TODO: support other repopulate strategies
        # TODO: generating whole random state probably overkill
        position_state = np.where(
            in_bounds.reshape((-1, 1)),
            position_state,
            random_pos,
        )
        type_state = np.where(
            in_bounds,
            type_state,
            random_type,
        )
        diag = scipy.sparse.diags(in_bounds.astype(int))
        bond_state = (diag * bond_state * diag).tolil()
        return {
            'position_state': position_state,
            'type_state': type_state,
            'bond_state': bond_state,
        }
