from collections import defaultdict
from itertools import product

import numpy as np


class Grid:
    def __init__(self, params):
        self._num_particle_types = params.particle.num_particle_types
        self._grid_size = params.grid.grid_size
        self._max_interaction_distance = params.interaction.max_interaction_distance

    def __call__(self, position_state, type_state, **kwargs):
        grid_array = np.floor_divide(position_state, self._grid_size)
        grid_remainder = np.remainder(position_state, self._grid_size)
        x_remainder = grid_remainder[:, 0]
        y_remainder = grid_remainder[:, 1]
        grid_adjacency = np.hstack((
            np.where(
                x_remainder < self._max_interaction_distance,
                -1,
                np.where(
                    x_remainder > self._grid_size - self._max_interaction_distance,
                    1,
                    0
                )
            )[:, np.newaxis],
            np.where(
                y_remainder < self._max_interaction_distance,
                -1,
                np.where(
                    y_remainder > self._grid_size - self._max_interaction_distance,
                    1,
                    0
                )
            )[:, np.newaxis],
        ))

        def dist(i, j):
            return np.linalg.norm(position_state[i] - position_state[j])

        # TODO: vectorize, or at least parallelize (here and elsewhere)
        grid_particles = defaultdict(list)
        for i, (row, col) in enumerate(grid_array):
            grid_particles[(row, col)].append(i)

        grid_state = []
        for i, (row, col) in enumerate(grid_array):
            x_dims = {0, grid_adjacency[i, 0]}
            y_dims = {0, grid_adjacency[i, 1]}

            adj = [[] for _ in range(self._num_particle_types)]
            for x_diff, y_diff in product(x_dims, y_dims):
                for p in grid_particles[row + x_diff, col + y_diff]:
                    d = dist(p, i)
                    if p != i and d < self._max_interaction_distance:
                        adj[type_state[p]].append((p, d))

            for i, a in enumerate(adj):
                adj[i] = sorted(a, key=lambda x: x[1])
            grid_state.append(adj)
        return {
            'grid_state': grid_state,
        }
