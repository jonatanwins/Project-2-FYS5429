import os
import matplotlib.pyplot as plt
from typing import List, Optional
from plot_utils.metrics import RunMetrics  # type: ignore -goofy linitng issue

def plot_metrics(metrics: RunMetrics, metric_names: List[str], title: Optional[str] = None, save_figure: bool = False, save_path: Optional[str] = None):
    """Plots and optionally saves the given metric(s) from a RunMetrics object.
    
    Args:
        metrics (RunMetrics): RunMetrics object containing the metrics.
        metric_names (List[str]): List of metric names to plot.
        title (Optional[str]): Title of the plot.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    for name in metric_names:
        if hasattr(metrics, name):
            metric_data = getattr(metrics, name)
            steps = metric_data["steps"]
            values = metric_data["values"]
            plt.plot(steps, values, label=name)
        else:
            print(f"Warning: '{name}' not found in the provided metrics.")

    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.title(title or "Metrics Plot")
    plt.legend()
    plt.grid(True)
    
    if save_figure:
        save_path = save_path or "../../figures"
        filename = title or "metrics_plot"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"))
    
    plt.show()


def plot_multiple_runs(log_dirs: List[str], metric_names: List[str], labels: Optional[List[str]] = None, title: Optional[str] = None, save_figure: bool = False, save_path: Optional[str] = None):
    """Plots and optionally saves the specified metric(s) across multiple runs.
    
    Args:
        log_dirs (List[str]): List of directories containing the logs.
        metric_names (List[str]): List of metric names to plot.
        labels (Optional[List[str]]): Labels for each run.
        title (Optional[str]): Title of the plot.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(log_dirs))]

    for log_dir, label in zip(log_dirs, labels):
        metrics = RunMetrics(log_dir)
        for name in metric_names:
            if hasattr(metrics, name):
                metric_data = getattr(metrics, name)
                steps = metric_data["steps"]
                values = metric_data["values"]
                plt.plot(steps, values, label=f"{label} - {name}")
            else:
                print(f"Warning: '{name}' not found in '{log_dir}'.")

    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.title(title or "Metrics Comparison")
    plt.legend()
    plt.grid(True)
    
    if save_figure:
        save_path = save_path or "../../figures"
        filename = title or "multiple_metrics_plot"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"))
    
    plt.show()


if __name__ == "__main__":
    plt.style.use("plot_settings.mplstyle")
    # Example usage
    log_dir = "../lorenz/checkpoints/KathleenReplicas/version_12"
    metrics = RunMetrics(log_dir)

    # Plot a single run
    plot_metrics(metrics, metric_names=["train/loss", "val/loss"], title="Loss Metrics")

    #plot_multiple_runs(log_dirs, metric_names=["train/loss", "val/loss"], labels=["Version 9", "Version 10"], title="Loss Comparison")
    #almost all of the out.tensorboard files are empty :(
