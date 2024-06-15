from tensorboard.backend.event_processing import event_accumulator

class RunMetrics:
    """
    A class to load and access metrics from a TensorBoard log directory.

    Attributes:
        log_dir (str): The path to the directory containing the TensorBoard log files.
        ea (EventAccumulator): An EventAccumulator object to load the events from the log directory.
        tags (list): A list of tags for the scalars in the log directory.
        metrics (dict): A dictionary containing the metrics loaded from the log directory.
    """
    def __init__(self, log_dir):
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

# Usage example
if __name__ == "__main__":
    log_dir = "../lorenz/checkpoints/KathleenReplicas/version_12"
    
    metrics = RunMetrics(log_dir)

   
    print(metrics.metrics.keys())
    for k in metrics.metrics:
        print(f"Printing {k}")
        print("============================================================================================================================================================")
        print(metrics.metrics[k]['values'])
        print("============================================================================================================================================================")
        print("\n\n\n")
