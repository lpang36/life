import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


class Display:
    def __init__(self, params):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)
        plt.show(block=False)
        plt.pause(0.1)
        self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)
        self._points = None
        self._attractor_gradient = None
        self._repeller_gradient = None
        self._lines = None

    def __call__(self, position_state, type_state, bond_state, attractor_gradient_state, repeller_gradient_state, **kwargs):
        plt.pause(0.01)
        bond_idx = np.reshape(
            np.array(bond_state.nonzero(), dtype=int).T, (-1, 2))

        if self._points is None:
            self._points = self._ax.scatter(
                position_state[:, 0], position_state[:, 1], c=type_state, cmap=matplotlib.colormaps['viridis'], s=5)
            self._attractor_gradient = self._ax.scatter(
                attractor_gradient_state[:,
                                         0], attractor_gradient_state[:, 1], color='black', s=100
            )
            self._repeller_gradient = self._ax.scatter(
                repeller_gradient_state[:,
                                        0], repeller_gradient_state[:, 1], color='red', s=100
            )
            lc = matplotlib.collections.LineCollection(
                position_state[bond_idx], color='red', linewidth=1)
            self._lines = self._ax.add_collection(lc)
        else:
            self._points.set_offsets(position_state)
            self._points.set_array(type_state)
            self._attractor_gradient.set_offsets(attractor_gradient_state)
            self._repeller_gradient.set_offsets(repeller_gradient_state)
            self._lines.set_segments(position_state[bond_idx])

        self._fig.canvas.restore_region(self._background)
        self._ax.draw_artist(self._points)
        self._ax.draw_artist(self._attractor_gradient)
        self._ax.draw_artist(self._repeller_gradient)
        self._ax.draw_artist(self._lines)
        self._fig.canvas.blit(self._ax.bbox)
