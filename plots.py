from matplotlib import pyplot as plt

class MetricPlots:
    """
    Gestisce la visualizzazione dei metrici durante l'addestramento e la valutazione.
    Crea una figura con sottotrame per diversi metrici e aggiorna i grafici in tempo reale.
    """
    
    def __init__(self, figure_num=1):
        """
         Inizializza la figura e le sottotrame per i metrici.
         Args:
            figure_num (int, opzionale): Numero della figura matplotlib. Defaults to 1.
        """
        fig = plt.figure(figure_num, figsize=(15, 5),  layout="constrained")
        self.plots = fig.subplot_mosaic(
            [
                ["total_profit", "total_reward", "roi"],
                ["delta_p", "step_reward", "wallet_value"]
            ]
        )
        
        self._set_plot_labels()
        plt.ion()

    def _set_plot_labels(self):
        """
        Imposta titoli e etichette per ciascuna sottotrama.
        """
        self.plots['total_profit'].set_title("Total Profit")
        self.plots['total_profit'].set_xlabel("Episode")
        self.plots['total_profit'].set_ylabel("Profit")
        
        self.plots['delta_p'].set_title("delta_p")
        self.plots['delta_p'].set_xlabel("Timesteps")
        self.plots['delta_p'].set_ylabel("Profit")

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
        """
        Aggiorna e visualizza i grafici dei metrici forniti.
        Args:
            show (bool, opzionale): Se True, mostra i grafici in modalità bloccante. Defaults to False.
            **kwargs: Metrici da tracciare con le rispettive liste di valori.
        """
        if not self.plots:
            self.init_plots()
        
        for metric, value in kwargs.items():
            if metric == 'step_reward':
                self.plots[metric].clear()
                self.plots[metric].scatter(range(len(value)), value, s=2**2)
            elif metric in self.plots.keys():
                self.plots[metric].clear()
                self.plots[metric].plot(value)

        self._set_plot_labels()
        plt.draw()
        plt.pause(0.001)

        if block:
            plt.show(block=True)
