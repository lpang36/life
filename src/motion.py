import numpy as np
import pandas as pd
import scipy


class Motion:
    def __init__(self, params):
        self._repulsion_distance = params.interaction.repulsion_distance
        self._repulsion_strength = params.interaction.repulsion_strength
        self._bond_strength = params.interaction.bond_strength
        self._max_motion = params.interaction.max_motion
        self._brownian_motion = params.interaction.brownian_motion
        self._num_particle_types = params.particle.num_particle_types
        self._attraction_matrix = np.array(params.particle.attraction_matrix)
        self._bond_matrix = np.array(params.particle.bond_matrix)
        self._gradient_strength = params.gradient.gradient_strength
        self._standard_deviation = params.gradient.standard_deviation

    def __call__(self, position_state, type_state, bond_state, grid_state, attractor_gradient_state, repeller_gradient_state, **kwargs):
        # repulsion and attraction
        columns = ["id", "x", "y"]
        repulsion_raw = []
        repulsion_dist = []
        attraction_raw = []
        attraction_dist = []
        for p1, p_adj in enumerate(grid_state):
            for type2, t_adj in enumerate(p_adj):
                for p2, d in t_adj:
                    if d < self._repulsion_distance:
                        repulsion_raw.append([
                            p1, p2, type2
                        ])
                        repulsion_dist.append(d)
                        attraction_raw.append([
                            p1, p2, type2
                        ])
                        attraction_dist.append(self._repulsion_distance)
                    else:
                        attraction_raw.append([
                            p1, p2, type2
                        ])
                        attraction_dist.append(d)

        # reshape to handle empty case
        repulsion_raw = np.reshape(np.array(repulsion_raw, dtype=int), (-1, 3))
        attraction_raw = np.reshape(
            np.array(attraction_raw, dtype=int), (-1, 3))
        repulsion_dist = np.array(repulsion_dist)
        attraction_dist = np.array(attraction_dist)
        # avoid divide by 0
        repulsion_dist += self._repulsion_distance

        def get_components(arr, dist, interactions_matrix):
            dest_state = position_state[arr[:, 0], :2]
            dest_type = type_state[arr[:, 0]]
            source_state = position_state[arr[:, 1], :2]
            interactions = interactions_matrix[arr[:, 2], dest_type]
            # inverse linear decay, since we're in 2d space
            dist_transform = 1 / dist
            return pd.DataFrame(np.hstack((
                arr[:, 0].reshape((-1, 1)),
                (source_state - dest_state) *
                np.reshape(interactions * dist_transform, (-1, 1))
            )), columns=columns)

        repulsion_components = get_components(
            repulsion_raw, repulsion_dist, -self._repulsion_strength * np.ones((self._num_particle_types, self._num_particle_types)))
        attraction_components = get_components(
            attraction_raw, attraction_dist, self._attraction_matrix)

        # bond
        bond_idx = np.reshape(
            np.array(bond_state.nonzero(), dtype=int).T, (-1, 2))
        dummy_ones = np.ones((len(bond_idx), 1), dtype=int)
        bond_components = get_components(np.hstack((
            bond_idx,
            dummy_ones,
        )), dummy_ones.T, self._bond_strength * np.ones((self._num_particle_types, self._num_particle_types)))

        def get_gradient(gx, gy):
            diff = position_state - np.array([gx, gy])
            dist = (diff ** 2).sum(axis=1) / self._standard_deviation
            gradient = (scipy.stats.norm.pdf(dist) *
                        self._gradient_strength)[:, np.newaxis]
            return diff * gradient

        gradient_components = np.hstack((
            np.arange(position_state.shape[0])[:, np.newaxis],
            np.zeros_like(position_state),
        ))
        for gx, gy in attractor_gradient_state:
            gradient_components[:, 1:] -= get_gradient(gx, gy)
        for gx, gy in repeller_gradient_state:
            gradient_components[:, 1:] += get_gradient(gx, gy)
        gradient_components = pd.DataFrame(
            gradient_components, columns=columns)

        forces = pd.concat(
            (repulsion_components, attraction_components, bond_components, gradient_components))
        forces = forces.groupby('id').sum().to_numpy()
        forces += np.random.normal(0, self._brownian_motion,
                                   position_state.shape)
        forces = forces.clip(-self._max_motion, self._max_motion)
        return {
            'position_state': position_state + forces,
        }
