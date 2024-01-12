from collections import defaultdict

import numpy as np


class Bond:
    # TODO: support multiple bonds
    def __init__(self, params):
        self._bond_map = defaultdict(list)
        # we arbitrarily only map for one reactant, which is probably fine
        # reactant a, reactant b, catalyst (-1 if none), probability
        for a, b, c, p in params.particle.bond_matrix:
            self._bond_map[int(a)].append((int(b), int(c), p))
        for k, v in self._bond_map.items():
            # sort by descending probability so we use the most likely reaction
            self._bond_map[k] = sorted(v, key=lambda x: -x[2])
        self._max_interaction_distance = params.interaction.max_interaction_distance

    def __call__(self, type_state, bond_state, grid_state, **kwargs):
        # exclude already bonded
        exclude = bond_state.sum(axis=1)
        bond_probs = []
        for i, type in enumerate(type_state):
            if exclude[i] > 0:
                bond_probs.append((0, 0))
                continue
            for b, c, p in self._bond_map[type]:
                if len(grid_state[i][b]) > 0 and (
                    c < 0 or len(grid_state[i][c]) > 0
                ):
                    # go through potential partners, exclude already bonded
                    found = False
                    for other, _ in grid_state[i][b]:
                        if exclude[other] == 0:
                            bond_probs.append((other, p))
                            found = True
                            break
                    if found:
                        break
            else:
                bond_probs.append((0, 0))
                continue

        n = len(exclude)
        bond_pairs = np.hstack(
            (np.arange(n)[:, np.newaxis], np.array(bond_probs), np.random.rand(n, 1)))
        for i, j, p, v in bond_pairs:
            if v < p:
                # TODO: could still cause two simultaneous bonds
                bond_state[int(i), int(j)] = 1
                bond_state[int(j), int(i)] = 1
        # TODO: can't use operations on lil matrix (or sparse in general)
        return {
            'bond_state': bond_state
        }
