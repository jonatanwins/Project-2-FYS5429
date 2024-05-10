import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from typing import List, Optional
import plot_settings  # type: ignore 




class RunMetrics:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            }
        )
        self.ea.Reload()
        self.tags = self.ea.Tags()["scalars"]
        self.metrics = self._fetch_metrics()

    def _fetch_metrics(self):
        metrics = {}
        for tag in self.tags:
            events = self.ea.Scalars(tag)
            steps, values = zip(*[(e.step, e.value) for e in events])
            metrics[tag] = {"steps": steps, "values": values}
        return metrics

    def __getattr__(self, name):
        """Dynamically access metrics as attributes."""
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def plot_metrics(metrics: RunMetrics, metric_names: List[str], title: Optional[str] = None):
    """Plots the given metric(s) from a RunMetrics object.

    Args:
        metrics (RunMetrics): RunMetrics object containing the metrics.
        metric_names (List[str]): List of metric names to plot.
        title (Optional[str]): Title of the plot.
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
    plt.title(title or "Metrics Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_runs(log_dirs: List[str], metric_names: List[str], labels: Optional[List[str]] = None, title: Optional[str] = None):
    """Plots the specified metric(s) across multiple runs.

    Args:
        log_dirs (List[str]): List of directories containing the logs.
        metric_names (List[str]): List of metric names to plot.
        labels (Optional[List[str]]): Labels for each run.
        title (Optional[str]): Title of the plot.
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
    plt.title(title or "Metrics Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    log_dir = "lorenz/checkpoints/Autoencoder/version_9"
    metrics = RunMetrics(log_dir)

    # Plot a single run
    plot_metrics(metrics, metric_names=["train_loss", "val_loss"], title="Loss Metrics")

    # Plot multiple runs
    log_dirs = [
        "../lorenz/checkpoints/Autoencoder/version_8",
        "../lorenz/checkpoints/Autoencoder/version_9"
    ]
    plot_multiple_runs(log_dirs, metric_names=["train_loss", "val_loss"], labels=["Version 9", "Version 10"], title="Loss Comparison")
