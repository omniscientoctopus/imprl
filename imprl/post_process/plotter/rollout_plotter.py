import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imprl.post_process.plotter.base_class import AbstractBaseClass


class AgentPlotter(AbstractBaseClass):
    def __init__(self, env, agent):
        super().__init__(env)

        self.agent = agent

    def get_sample_rollout(self):
        # data to be collected
        data = {
            "time": np.arange(0, self.time_horizon + 1),
            "true_states": np.empty((self.time_horizon + 1, self.num_components)),
            "beliefs": np.empty(
                (self.time_horizon + 1, self.num_damage_states, self.num_components)
            ),
            "actions": np.ones((self.time_horizon, self.num_components), dtype=int)
            * -1,
            "rewards": np.empty(self.time_horizon),
            "cost_penalties": np.empty(self.time_horizon),
            "cost_inspections": np.empty(self.time_horizon),
            "cost_replacements": np.empty(self.time_horizon),
            "failure_prob": np.empty(self.time_horizon + 1),
            "failure_timepoints": np.zeros(self.time_horizon + 1),
            "episode_cost": 0,
        }

        done = False
        episode_reward = 0
        time = 0

        observation = self.env.reset()
        _, system_belief = observation
        data["beliefs"][0, :, :] = system_belief

        while not done:
            data["true_states"][time, :] = self.env.damage_state

            # select action
            action = self.agent.select_action(observation, training=False)

            next_observation, reward, done, info = self.env.step(action)

            # update belief
            _, system_belief = next_observation
            data["beliefs"][time + 1, :, :] = system_belief
            data["actions"][time, :] = action
            data["rewards"][time] = reward
            data["cost_penalties"][time] = info["reward_penalty"]
            data["cost_inspections"][time] = info["reward_inspection"]
            data["cost_replacements"][time] = info["reward_replacement"]

            # note system failure timepoints
            if info["reward_penalty"] != 0:
                data["failure_timepoints"][time] = 1

            # update observation
            observation = next_observation

            # update episode reward
            episode_reward += reward

            # update time
            time += 1

        data["episode_cost"] = -episode_reward
        data["true_states"][time, :] = self.env.damage_state

        return data

    def plot(self, save_fig_kwargs=None):
        # get data from agent rollout
        data = self.get_sample_rollout()

        self.data = data

        # get base plot from Plotter
        fig, ax_dict, legend_handles, barplot = super()._setup_plot()

        _y_action = 2

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            # belief
            colorbar = ax.pcolormesh(
                data["time"],
                np.arange(self.num_damage_states),
                data["beliefs"][:, :, c].T,
                shading="nearest",
                cmap="viridis_r",  # _r for reversed
                alpha=0.2,
                vmin=0,
                vmax=1,
                edgecolors="face",
            )

            # true state
            (h_true_state,) = ax.plot(
                data["time"],
                data["true_states"][:, c],
                "-",
                label="true state",
                color="tab:green",
                markersize=2,
                alpha=0.8,
            )

            ax.set_yticks([0, 1, 2, 3])
            ax.set_ylim([-0.5, 4.05])
            ax.set_ylabel("true state", fontsize=14)
            ax.set_xticks([0, 10, 20, 30, 40, 50])
            ax.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize=14)

            # draw vertical lines when system fails
            if data["failure_timepoints"].sum() > 0:
                ax.vlines(
                    np.where(data["failure_timepoints"]),
                    -1,
                    3.5,
                    label="system-failure",
                    color="red",
                    alpha=1,
                )

            # do nothing
            _x = np.where(data["actions"][:, c] == 0)
            ax.plot(_x, _y_action, ".", label="do-nothing", markersize=5, color="gray")

            # repair
            _x = np.where(data["actions"][:, c] == 1)
            ax.plot(
                _x, _y_action, "s", label="repair", markersize=5, color="darkviolet"
            )

            # inspect
            _x = np.where(data["actions"][:, c] == 2)
            ax.plot(_x, _y_action, ">", label="inspect", markersize=5, color="orange")

            ax.set_ylabel("damage-state", fontsize=13)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(False)

        # update legend_handles
        legend_handles += [h_true_state]
        if data["failure_timepoints"].sum() > 0:
            legend_handles += [
                Line2D([], [], color="red", label="system-failure", alpha=1)
            ]
        fig.legend(handles=legend_handles, loc=(0.83, 0.1), fontsize=14)

        # colorbar for belief
        cax = ax_dict["3"].inset_axes([1.05, 0.0, 0.02, 0.8])
        fig.colorbar(colorbar, label="belief", cax=cax, location="right", shrink=5)

        ## Bar plot
        total_inspect = -data["cost_inspections"].sum()
        total_inspect = (
            total_inspect if total_inspect > 0 else 1e-10
        )  # avoid warning in pie chart
        total_replace = -data["cost_replacements"].sum()
        total_penalty = -data["cost_penalties"].sum()
        total_cost = total_inspect + total_replace + total_penalty

        _all = (
            np.array([total_replace, total_inspect, total_penalty]) * 100 / total_cost
        )

        # update bar plot
        for b, bar in enumerate(barplot):
            bar.set_width(_all[b])

        ax_dict["B"].set_title(f"Episode cost: {data['episode_cost']:.3f}", fontsize=14)

        fig.tight_layout()
        plt.show()

        if save_fig_kwargs is not None:
            fig.savefig(**save_fig_kwargs)
