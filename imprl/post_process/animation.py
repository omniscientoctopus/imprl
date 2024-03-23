""" 

Click on the figure to pause, and press any key to continue.

"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms

sns.set_theme(style="white", palette="muted")


class AnimatedRollout:
    def __init__(self, env, agent) -> None:
        self.agent = agent

        self.env = env
        self.time_horizon = env.time_horizon
        self.num_components = env.n_components
        self.num_damage_states = env.n_damage_states

        # collect data
        self.rollout()

        # initialize figure
        self.setup_plot()

    def setup_plot(
        self,
    ):
        # initialize figure
        # setup using mosaic
        self.fig, self.axs = plt.subplot_mosaic(
            [
                ["1", "1", ".", ".", "."],
                [".", "3", "3", ".", "."],
                ["2", "2", ".", "5", "5"],
                [".", "4", "4", ".", "."],
            ],
            layout="constrained",
            figsize=(10, 6),
        )

        # add labels to subplots
        for label, ax in self.axs.items():
            # label physical distance in and down:
            trans = mtransforms.ScaledTranslation(
                10 / 72, -5 / 72, self.fig.dpi_scale_trans
            )
            ax.text(
                0.0,
                1.0,
                label,
                transform=ax.transAxes + trans,
                fontsize="medium",
                verticalalignment="top",
                fontfamily="serif",
                bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0),
            )

        _num_legend_items = 0
        saved_handles = None
        saved_labels = None

        # line objects for each line
        # (compo-risk, failure-risk, system failues, do nothing, inspect, repair)
        # num_components x 6
        self.lines = []

        # loop over components
        for label, ax in self.axs.items():
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-0.5, 50.5])
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel(r"$p_f$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.grid()

            ax2 = ax.twinx()
            ax2.set_yticks([0, 1, 2])
            ax2.tick_params(right=False, labelright=False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)

            # component risk
            h_component_risk = ax.plot(
                [],
                [],
                "-",
                label="failure-risk",
                color="crimson",
                markersize=2,
                alpha=1,
            )

            # system risk
            h_sys_risk = ax.plot(
                [],
                [],
                "-",
                label="system-failure-risk",
                color="tab:pink",
                markersize=2,
                alpha=0.5,
            )

            # system risk
            h_failure = ax.plot(
                [],
                [],
                "X",
                label="system-failure",
                color="indigo",
                markersize=5,
                alpha=0.4,
            )

            # do nothing
            h_do_nothing = ax2.plot(
                [], [], ".", label="do-nothing", markersize=5, color="gray"
            )

            # replace
            h_replace = ax2.plot(
                [], [], "s", label="replace", markersize=5, color="darkviolet"
            )

            # inspect
            h_inspect = ax2.plot(
                [], [], ">", label="inspect", markersize=7, color="orange"
            )

            # udpate list
            self.lines.append(h_component_risk[0])
            self.lines.append(h_sys_risk[0])
            self.lines.append(h_failure[0])

            self.lines.append(h_do_nothing[0])
            self.lines.append(h_replace[0])
            self.lines.append(h_inspect[0])

            # (hack to get legend labels and handles)
            # check if labels have all elements
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if len(by_label.keys()) > _num_legend_items:
                saved_handles = handles
                saved_labels = labels
                _num_legend_items = len(by_label.keys())

        # actions legend
        by_label = dict(zip(saved_labels, saved_handles))
        by_label.update({f"component-risk": h_component_risk[0]})
        by_label.update({f"system-risk": h_sys_risk[0]})
        by_label.update({f"system failure": h_failure[0]})
        self.fig.legend(by_label.values(), by_label.keys(), loc=(0.55, 0.8))

    def run(
        self,
    ):
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
        # update lines
        for c in range(self.num_components):
            # component risk
            self.lines[6 * c].set_data(
                self.data["time"][: frame + 1], self.data["beliefs"][: frame + 1, -1, c]
            )

            # system risk
            self.lines[6 * c + 1].set_data(
                self.data["time"][: frame + 1], self.data["failure_prob"][: frame + 1]
            )

            # system failure
            _x = np.where(self.data["failure_timepoints"][: frame + 1] == 1)[0]
            self.lines[6 * c + 2].set_data(_x, 0.8)

            # do nothing
            _x = np.where(self.data["actions"][: frame + 1, c] == 0)
            self.lines[6 * c + 3].set_data(_x, 1)

            # replace
            _x = np.where(self.data["actions"][: frame + 1, c] == 1)
            self.lines[6 * c + 4].set_data(_x, 1)

            # inspect
            _x = np.where(self.data["actions"][: frame + 1, c] == 2)
            self.lines[6 * c + 5].set_data(_x, 1)

        return self.lines

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def rollout(
        self,
    ):
        # data to be collected
        self.data = {
            "time": np.arange(0, self.time_horizon + 1),
            "beliefs": np.empty(
                (self.time_horizon + 1, self.num_damage_states, self.num_components)
            ),
            "actions": np.ones((self.time_horizon, self.num_components), dtype=int)
            * -1,
            "rewards": np.empty(self.time_horizon),
            "penalties": np.empty(self.time_horizon),
            "inspections": np.empty(self.time_horizon),
            "replacements": np.empty(self.time_horizon),
            "failure_prob": np.empty(self.time_horizon + 1),
            "failure_timepoints": np.ones(self.time_horizon + 1) * -1,
            "episode_cost": 0,
        }

        done = False
        episode_reward = 0
        time = 0

        observation = self.env.reset()
        _, system_belief = observation
        self.data["beliefs"][0, :, :] = system_belief

        while not done:
            # select action
            action = self.agent.select_action(observation, training=False)

            next_observation, reward, done, info = self.env.step(action)

            self.data["actions"][time] = action
            self.data["rewards"][time] = reward
            self.data["penalties"][time] = info["cost_penalty"]
            self.data["inspections"][time] = info["cost_inspection"]
            self.data["replacements"][time] = info["cost_replacement"]
            self.data["failure_prob"][time + 1] = 1 - info["system_reliability"]

            # note system failure timepoints
            if info["cost_penalty"] != 0:
                self.data["failure_timepoints"][time + 1] = 1

            # update belief
            _, system_belief = next_observation
            self.data["beliefs"][time + 1, :, :] = system_belief

            # update observation
            observation = next_observation

            # update episode reward
            episode_reward += self.env.discount_factor**time * reward

            # update time
            time += 1

        self.data["episode_cost"] = -episode_reward
