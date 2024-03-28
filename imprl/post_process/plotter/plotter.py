import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Plotter:
    def __init__(self, env) -> None:
        self.env = env
        self.time_horizon = env.time_horizon
        self.num_components = env.n_components
        self.num_damage_states = env.n_damage_states

    def _plot(self):
        # check if data has obs or beliefs
        sns.set_theme(style="white", palette="muted")

        mosaic = """
            1155
            2244
            B33.
            """

        fig = plt.figure(layout="constrained", figsize=(12, 6))
        ax_dict = fig.subplot_mosaic(mosaic)

        plt.rcParams.update(
            {
                "axes.titlesize": "medium",
                "axes.labelsize": "large",
            }
        )

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            ## Plot agent actions
            ax2 = ax.twinx()
            ax2.set_yticks([0, 1, 2])

            ax2.set_yticks([0, 1, 2])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-0.5, 50.5])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("time", fontsize=15)
            ax.set_title(f"Component {c+1}", weight="bold", fontsize=16)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax2.spines[["top", "right", "left"]].set_visible(False)
            ax.grid()

            ax2.tick_params(right=False, labelright=False)

        # create legend handles
        legend_handles = [
            Line2D([], [], marker=".", markersize=5, color="gray", label="do-nothing"),
            Line2D(
                [], [], marker="s", markersize=5, color="darkviolet", label="repair"
            ),
            Line2D([], [], marker=">", markersize=7, color="orange", label="inspect"),
        ]

        labels = ["repair", "inspect", "failure"]
        colors = ["darkviolet", "orange", "lightcoral"]

        barplot = ax_dict["B"].barh(labels, [0] * 3, color=colors, height=0.4)
        ax_dict["B"].set_xlim([0, 100])
        ax_dict["B"].set_xticks([0, 25, 50, 75, 100])
        ax_dict["B"].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        return fig, ax_dict, legend_handles, barplot

    def get_sample_rollout(self):
        NotImplementedError
