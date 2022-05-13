# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from agml.viz.tools import show_when_allowed


class Arrow3D(FancyArrowPatch):
    """Draws a 3-dimensional arrow using the given coordinates."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._coords = xs, ys, zs

    def draw(self, renderer):
        xs, ys, zs = self._coords
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


@show_when_allowed
def plot_synthetic_camera_positions(positions, lookat):
    """Plots camera perspectives: positions and lookat vectors.

    This method is used to plot camera perspectives from a provided
    set of positional coordinates and lookat vector coordinates. This
    can be used to visualize the perspectives of cameras when generating
    synthetic agricultural data using Helios.

    The output of the method `agml.synthetic.generate_camera_positions`
    can be piped directly into this method to get a plot straight from
    environment parameters.

    Parameters
    ----------
    positions : list
        A list of 3-d coordinates indicating the positions of the
        cameras whose perspectives are to be plotted.
    lookat : list
        A list of lookat vector coordinates.

    Returns
    -------
    The matplotlib figure with the plot.
    """
    # Creat ethe figure.
    plt.figure(figsize = (7, 7))
    ax = plt.axes(projection = '3d')
    ax.plot3D(0, 0, 1, 'r*', label = 'Canopy Origin')
    for i in range(len(positions)):
        ax.plot3D(positions[i][0], positions[i][1], positions[i][2], 'o',
                  label = 'Camera ' + str(i))
        arw = Arrow3D([positions[i][0], lookat[i][0]],
                      [positions[i][1], lookat[i][1]],
                      [positions[i][2], lookat[i][2]],
                      arrowstyle = "->", color = "purple",
                      lw = 1, mutation_scale = 25)
        ax.add_artist(arw)
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.05),
              ncol = 4, fancybox = True, shadow = True)
    fig = plt.gcf()
    return fig




