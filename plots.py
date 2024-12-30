from matplotlib import pyplot as plt

class MetricPlots:
    def __init__(self, figure_num=1):
        fig = plt.figure(figure_num, figsize=(15, 5),  layout="constrained")
        self.plots = fig.subplot_mosaic(
            [
                ["total_profit", "total_reward", "roi"],
                ["step_profit", "step_reward", "wallet_value"]
            ]
        )
        
        self._set_plot_labels()
        plt.ion()

    def _set_plot_labels(self):
        self.plots['total_profit'].set_title("Total Profit")
        self.plots['total_profit'].set_xlabel("Episode")
        self.plots['total_profit'].set_ylabel("Profit")
        
        self.plots['step_profit'].set_title("Step Profit")
        self.plots['step_profit'].set_xlabel("Timesteps")
        self.plots['step_profit'].set_ylabel("Profit")

        self.plots['total_reward'].set_title("Total Reward")
        self.plots['total_reward'].set_xlabel("Episode")
        self.plots['total_reward'].set_ylabel("Reward")

        self.plots['step_reward'].set_title("Step Reward")
        self.plots['step_reward'].set_xlabel("Timesteps")
        self.plots['step_reward'].set_ylabel("Reward")

        self.plots['roi'].set_title("ROI")
        self.plots['roi'].set_xlabel("Episode")
        self.plots['roi'].set_ylabel("ROI")

        self.plots['wallet_value'].set_title("Wallet Value")
        self.plots['wallet_value'].set_xlabel("Episode")
        self.plots['wallet_value'].set_ylabel("Value")

    def plot_metrics(self, show=False, **kwargs):
        if not self.plots:
            self.init_plots()
        
        for metric, value in kwargs.items():
            self.plots[metric].clear()
            if metric == 'step_reward':
                self.plots[metric].scatter(range(len(value)), value, s=2**2)
            else:
                self.plots[metric].plot(value)

        self._set_plot_labels()
        plt.draw()
        plt.pause(0.01)

        if show:
            plt.show(block=True)
