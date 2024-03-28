import numpy as np
import matplotlib.pyplot as plt
from imprl.post_process.plotter.plotter import Plotter


class HeuristicPlotter(Plotter):
    def __init__(self, env, policy):
        super().__init__(env)

        self.policy = policy

    def get_sample_rollout(self):
        # data to be collected
        data = {
            "time": np.arange(0, self.time_horizon + 1),
            "true_states": np.empty((self.time_horizon + 1, self.num_components)),
            "obs": np.ones((self.time_horizon + 1, self.num_components)) * -1,
            "actions": np.ones((self.time_horizon, self.num_components), dtype=int)
            * -1,
            "inspections": np.ones((self.time_horizon), dtype=int) * -1,
            "rewards": np.empty(self.time_horizon),
            "cost_penalties": np.zeros(self.time_horizon),
            "cost_inspections": np.zeros(self.time_horizon) * -1,
            "cost_replacements": np.zeros(self.time_horizon),
            "failure_prob": np.empty(self.time_horizon + 1),
            "failure_timepoints": np.full(self.time_horizon + 1, False, dtype=bool),
            "episode_cost": 0,
        }

        done = False
        episode_reward = 0
        time = 0

        _ = self.env.reset()
        observation = self.env.info["observation"]

        while not done:
            _, state = self.env.env.state
            data["true_states"][time, :] = state

            # compute actions using policy
            action, inspected_components = self.policy(time, observation)

            # step in the environment
            _, reward, done, info = self.env.step(action)

            data["actions"][time] = action

            # update episode reward
            data["rewards"][time] = reward
            episode_reward += reward

            # if inspection took place
            if inspected_components:
                _inspection_cost = (
                    self.env.cost_action[inspected_components, 2].sum()
                )
                episode_reward += _inspection_cost
                data["cost_inspections"][time] = _inspection_cost
                data["inspections"][time] = 1
                data["obs"][time, :] = observation  # current observation
            else:
                data["obs"][time, :] = -1

            observation = info["observation"]  # update observation

            data["cost_penalties"][time] = info["cost_penalty"]
            data["cost_replacements"][time] = (
                info["cost_replacement"]
            )
            data["failure_prob"][time + 1] = 1 - info["system_reliability"]

            # note system failure timepoints
            if info["cost_penalty"] != 0:
                data["failure_timepoints"][time] = True

            # update observation
            observation = info["observation"]  # update observation

            # update time
            time += 1

        data["episode_cost"] = -episode_reward

        _, state = self.env.env.state
        data["true_states"][time, :] = state

        return data

    def plot(self, save_fig_kwargs=None):
        # get data from agent rollout
        data = self.get_sample_rollout()
        self.data = data  # for analysis

        # get base plot from Plotter
        fig, ax_dict, legend_handles, barplot = super().plot()

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            # system failure risk
            (h_sys_failure_risk,) = ax.plot(
                data["time"],
                data["failure_prob"],
                "-",
                label="system failure risk",
                color="tab:pink",
                markersize=2,
                alpha=0.5,
            )

            # inspect
            _x = np.where(data["inspections"] == 1)
            ax.plot(_x, 0.5, ">", label="inspect", markersize=7, color="orange")

            # do nothing
            _x = np.where(data["actions"][:, c] == 0)
            ax.plot(_x, 0.5, ".", label="do-nothing", markersize=5, color="gray")

            # repair
            _x = np.where(data["actions"][:, c] == 1)
            ax.plot(_x, 0.5, "s", label="repair", markersize=5, color="darkviolet")

            # draw vertical lines when system fails
            if data["failure_timepoints"].any():
                ax.vlines(
                    np.where(data["failure_timepoints"]),
                    -1,
                    4,
                    label="system-failure",
                    linestyles="dashdot",
                    color="indigo",
                    alpha=0.4,
                )

            # ax.set_ylim([-0.15, 3.05])
            # ax.set_yticks([0, 1, 2, 3])
            # ax.set_ylabel("observation")

            # true state
            ax2 = ax.twinx()

            # true state
            (h_true_state,) = ax2.plot(
                data["time"],
                data["true_states"][:, c],
                "-",
                label="true state",
                color="tab:blue",
                markersize=2,
                alpha=0.5,
            )

            ## Plot agent observations
            (h_obs,) = ax2.plot(
                data["time"],
                data["obs"][:, c],
                "x-",
                label="obs",
                color="green",
                markersize=7,
                alpha=1,
                linewidth=1,
            )

            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_ylim([-0.15, 3.05])
            ax2.set_ylabel("state", fontsize=14)

        # update legend_handles
        legend_handles += [h_obs, h_sys_failure_risk, h_true_state]
        fig.legend(handles=legend_handles, loc=(0.8, 0.1))

        ## Bar plot
        total_inspect = -data["cost_inspections"].sum()
        total_replace = -data["cost_replacements"].sum()
        total_penalty = -data["cost_penalties"].sum()
        total_cost = total_inspect + total_replace + total_penalty

        _all = (
            np.array([total_replace, total_inspect, total_penalty]) * 100 / total_cost
        )

        # update bar plot
        for b, bar in enumerate(barplot):
            bar.set_width(_all[b])

        ax_dict["B"].set_title(
            f"Episode cost: {data['episode_cost']:.3f}", fontsize=14
        )  # Set episode cost as the title

        # axbox = ax_dict["4"].get_position()

        # ax_dict["4"].text(
        #     0.1,
        #     0.1,
        #     f"Inspection interval (∆T*): {self.policy.inspection_interval}",
        #     fontsize=10,
        #     horizontalalignment="right",
        #     verticalalignment="bottom",
        #     transform=ax_dict["4"].transAxes,
        #     fontweight="bold",
        # )
        # ax_dict["4"].text(
        #     0.6,
        #     0.6,
        #     f"sub-policy: {self.policy.policy}",
        #     fontsize=10,
        #     horizontalalignment="right",
        #     verticalalignment="bottom",
        #     transform=ax_dict["4"].transAxes,
        #     fontweight="bold",
        # )

        plt.tight_layout()  # Add tight layout

        if save_fig_kwargs is not None:
            fig.savefig(**save_fig_kwargs)

        plt.show()
