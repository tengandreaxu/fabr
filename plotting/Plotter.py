import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)


# *********************
# Plots Palette and Styles
# *********************
params = {
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "legend.fontsize": 14,
}
pylab.rcParams.update(params)


@dataclass
class Plotter:
    def __init__(self):
        self.logger = logging.getLogger("Plotter")
        self.colors = ["black", "brown"]
        self.line_styles = ["solid", "dashed"]

    def scatter_plot(self, x: np.ndarray, y: np.ndarray, file_name: str):

        plt.scatter(x, y)
        plt.ylabel("Predicted Value")
        plt.xlabel("True Value")
        m, b = np.polyfit(x, y, 1)

        plt.plot(x, m * x + b, "r")
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def plot_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
        file_name: str,
        title: str,
        xlogscale: Optional[bool] = False,
        label: Optional[str] = "",
    ):
        plt.plot(x, y, color="black", label=label, marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if xlogscale:
            plt.xscale("log")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def plot_curves_shared_x(
        self,
        x: np.ndarray,
        y1: list,
        y2: list,
        x_label: str,
        y1_label: str,
        y2_label: str,
        labels: list,
        file_name: str,
    ):
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        for y, color, linestyle, label in zip(
            y1, self.colors, self.line_styles, labels
        ):

            ax1.plot(x, y, color=color, linestyle=linestyle, label=label)
        for y, color, linestyle, label in zip(
            y2, self.colors, self.line_styles, labels
        ):

            ax2.plot(x, y, color=color, linestyle=linestyle, label=label)
        ax1.set_ylabel(y1_label)
        ax2.set_ylabel(y2_label)
        ax1.set_xlabel(x_label)
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        fig.savefig(file_name)
        plt.close()

    def plot_multiple_curves_from_dataframe_columns(
        self,
        df: pd.DataFrame,
        lock_column: str,
        y: str,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        vline: Optional[int] = 0,
        vline_label: Optional[str] = "",
        title: Optional[str] = "",
        ylim=[],  # Optional,# [Tuple[int, int]] = [],
        grid: Optional[bool] = False,
    ):
        values_ = df[lock_column].unique()
        for value in values_:
            sub_df = df[df[lock_column] == value].copy()
            plt.plot(sub_df[x], sub_df[y], label=value)
        plt.xlabel(xlabel)
        plt.grid(grid)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.axvline(vline, linestyle="dashed", color="black", label=vline_label)

        if len(ylim) > 0:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.legend()
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_curves_lists(
        self,
        ys: list,  # [Tuple[str, list]],
        xs: list,
        ylabel: str,
        xlabel: str,
        file_name: str,
        title: str,
        show: Optional[bool] = False,
        grid: Optional[bool] = False,
        remove_top_spine: Optional[bool] = False,
        remove_right_spine: Optional[bool] = False,
        aspect_ratio: Optional[float] = 0.0,
    ):

        for y in ys:
            lists, label_ = y
            plt.plot(xs, lists, label=label_)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(title)
        plt.grid(grid)
        plt.tight_layout()
        ax = plt.gca()
        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        if remove_right_spine:
            ax.spines["right"].set_visible(False)

        if aspect_ratio > 0:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect_ratio)
        if show:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_multiple_curves_from_dataframe_given_columns(
        self,
        df: pd.DataFrame,
        columns: list,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        suptitle: Optional[bool] = "",
        labels: Optional[list] = [],
        title: Optional[float] = "",
        vline: Optional[float] = 0.0,
        hline: Optional[float] = 0.0,
        hline_label: Optional[float] = "",
        from_index: Optional[bool] = False,
        vline_label: Optional[str] = "",
        grid: Optional[bool] = False,
        remove_top_spine: Optional[bool] = False,
        remove_right_spine: Optional[bool] = False,
        aspect_ratio: Optional[float] = 0.0,
    ):

        if len(labels) > 0:
            for y, label in zip(columns, labels):
                if from_index:

                    plt.plot(
                        df[~df[y].isna()][y].index,
                        df[~df[y].isna()][y].values,
                        label=label,
                    )
                else:
                    plt.plot(df[x], df[y], label=label)
        else:
            for y in columns:
                if from_index:
                    plt.plot(df.index, df[y], label=y)
                else:
                    plt.plot(df[x], df[y], label=y)
        plt.xlabel(xlabel)

        plt.grid(grid)
        plt.suptitle(suptitle)
        plt.title(title)
        if vline > 0:
            plt.axvline(vline, linestyle="dashed", color="black", label=vline_label)
        if hline > 0:
            plt.axhline(hline, linestyle="dotted", color="red", label=hline_label)

        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        ax = plt.gca()
        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        if remove_right_spine:
            ax.spines["right"].set_visible(False)

        if aspect_ratio > 0:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect_ratio)
        ax.set_rasterized(True)
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_curves_from_dataframes_given_columns(
        self,
        dfs: list,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        colors: list = [],
        title: Optional[float] = "",
        vline: Optional[float] = 0.0,
        from_index: Optional[bool] = False,
        vline_label: Optional[str] = "",
        xlogscale: Optional[bool] = False,
        grid: Optional[bool] = False,
        remove_top_spine: Optional[bool] = False,
        remove_right_spine: Optional[bool] = False,
        aspect_ratio: Optional[float] = 0.0,
    ):
        """dfs is a list of tuple (df, linestyle)"""
        if len(colors) > 0:
            for df_linestyle, color in zip(dfs, colors):
                df, linestyle = df_linestyle
                for y in df.columns:
                    if from_index:
                        plt.plot(
                            df.index, df[y], linestyle=linestyle, color=color, label=y
                        )
                    else:
                        plt.plot(
                            df[x], df[y], linestyle=linestyle, color=color, label=y
                        )
        else:
            for df, linestyle in dfs:
                for y in df.columns:
                    if from_index:
                        plt.plot(df.index, df[y], linestyle=linestyle, label=y)
                    else:
                        plt.plot(df[x], df[y], linestyle=linestyle, label=y)
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid(grid)
        plt.title(title)
        if vline > 0:
            plt.axvline(vline, linestyle="dashed", color="black", label=vline_label)
        plt.ylabel(ylabel)
        if xlogscale:
            plt.xscale("log")
        plt.tight_layout()
        ax = plt.gca()
        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        if remove_right_spine:
            ax.spines["right"].set_visible(False)

        if aspect_ratio > 0:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect_ratio)
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_curves_from_dataframes_given_columns_same_legend(
        self,
        dfs: list,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        colors: list,
        labels: list,
        linestyles: list,
        title: Optional[float] = "",
        vline: Optional[float] = 0.0,
        from_index: Optional[bool] = False,
        vline_label: Optional[str] = "",
        xlogscale: Optional[bool] = False,
        grid: Optional[bool] = False,
        remove_top_spine: Optional[bool] = False,
        remove_right_spine: Optional[bool] = False,
        aspect_ratio: Optional[float] = 0.0,
    ):

        for df, color, label_ in zip(dfs, colors, labels):
            for y, linestyle, label in zip(df.columns, linestyles, label_):
                if from_index:
                    plt.plot(
                        df.index, df[y], linestyle=linestyle, label=label, color=color
                    )
                else:
                    plt.plot(
                        df[x], df[y], linestyle=linestyle, label=label, color=color
                    )
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid(grid)
        plt.title(title)
        if vline > 0:
            plt.axvline(vline, linestyle="dashed", color="black", label=vline_label)
        plt.ylabel(ylabel)
        if xlogscale:
            plt.xscale("log")
        plt.tight_layout()
        ax = plt.gca()
        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        if remove_right_spine:
            ax.spines["right"].set_visible(False)

        if aspect_ratio > 0:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect_ratio)
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_dataframes(
        self,
        dfs: list,
        y: str,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        title: Optional[str] = "",
        vline: Optional[float] = 0.0,
        from_index: Optional[bool] = False,
        vline_label: Optional[str] = "",
    ):
        for df, label, linestyle, linewidth, marker in dfs:
            if from_index:
                plt.plot(
                    df.index,
                    df[y],
                    label=label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                )
            else:
                plt.plot(
                    df[x],
                    df[y],
                    label=label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                )
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid(True)
        plt.title(title)
        plt.axvline(vline, linestyle="dashed", color="black", label=vline_label)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def plot_quad_grid_from_dataframe(
        self,
        df: pd.DataFrame,
        lock_columns: list,
        x: str,
        ylabel: str,
        xlabel: str,
        file_name: str,
        title: Optional[str] = "",
        vline: Optional[float] = 0.0,
        vline_label: Optional[str] = "",
        xscalelog: Optional[bool] = False,
    ):

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, column in zip(axes, lock_columns):
            for nu in df.nu.unique():
                sub_curve = df[(df.nu == nu)]
                ax.plot(sub_curve[x], sub_curve[column], label=f"nu={nu}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.title.set_text(f"z = {column}")

                if vline > 0:
                    ax.axvline(vline, linestyle="dotted", color="black")
                if xscalelog:
                    ax.set_xscale("log")
                ax.legend()
                ax.grid(True)

        fig.suptitle(title, fontsize=22, fontweight="bold")
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(file_name, dpi=300)
        plt.close()
