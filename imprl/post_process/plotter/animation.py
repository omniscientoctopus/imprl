""" 

Click on the figure to pause, and press any key to continue.

"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

from imprl.post_process.plotter.agent_plotter import AgentPlotter

sns.set_theme(style="white", palette="muted")


class AnimatedRollout(AgentPlotter):
    def __init__(self, env, agent) -> None:
        super().__init__(env, agent)

        # initialize figure
        self.fig, self.ax_dict, legend_handles, barplot = self._plot()

        # line objects for each line
        # [belief, true state]
        self.lines = []

        for c in range(self.num_components):

            ax = self.ax_dict[f"{c+1}"]

            # plot true states
            (line,) = ax.plot([], [], label="true state", color="tab:green")
            self.lines.append(line)

            ax.set_yticks(np.arange(self.num_damage_states))
            ax.set_ylim([-0.5, 4.05])
            ax.set_ylabel("true state", fontsize=14)

            # plot do-nothing
            h_do_nothing = ax.plot(
                [],
                [],
                ".",
                markersize=5,
                color="gray",
                label="do-nothing",
            )
            self.lines.append(h_do_nothing[0])

            # plot replace
            h_repair = ax.plot(
                [],
                [],
                "s",
                markersize=5,
                color="darkviolet",
                label="repair",
            )
            self.lines.append(h_repair[0])

            # plot inspect
            h_inspect = ax.plot(
                [],
                [],
                ">",
                markersize=5,
                color="orange",
                label="inspect",
            )
            self.lines.append(h_inspect[0])


    def run(self):

        # get sample rollout
        self.data = self.get_sample_rollout()

        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.time_horizon + 1,
            init_func=self.init,
            blit=True,
            interval=200,
            repeat=True,
        )

        # pause animation on click or key press
        self.paused = False
        self.fig.canvas.mpl_connect("button_press_event", self.toggle_pause)
        self.fig.canvas.mpl_connect("key_press_event", self.toggle_pause)

        plt.show()

    def init(
        self,
    ):
        for line in self.lines:
            line.set_data([], [])  # Clear the lines

        return self.lines

    def update(self, frame):

        n = 4

        for c in range(self.num_components):

            _x = self.data["time"][: frame + 1]

            # update true states
            _y = self.data["true_states"][: frame + 1, c]
            self.lines[n * c].set_data(_x, _y)

            # update do-nothing
            self.lines[n * c + 1].set_data(
                np.where(self.data["actions"][: frame + 1, c] == 0), 1
            )

            # update repair
            self.lines[n * c + 2].set_data(
                np.where(self.data["actions"][: frame + 1, c] == 1), 1
            )

            # update inspect
            self.lines[n * c + 3].set_data(
                np.where(self.data["actions"][: frame + 1, c] == 2), 1
            )

        return self.lines

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
