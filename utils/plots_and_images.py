import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Plot:
    def __init__(
            self,
            dataframe,
            order: list,
            out_dir: str,
            show: bool,
            mode: str):

        self.df = dataframe
        self.order = order
        self.out_dir = out_dir
        self.show = show
        self.mode = mode

    def show_count_barplot(self, x):

        ax = sns.countplot(data=self.df, x=x, order=self.order)

        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01))

        plt.title(f'Count plot - {self.mode}')
        if self.out_dir:
            plt.savefig(f'{self.out_dir}/{self.mode}/CatCount_{self.mode}.png')
        if bool(self.show):
            plt.show()
        plt.close()

    def scatter_plot_per_category(self, col, hue, col_wrap):

        grid = sns.FacetGrid(self.df,
                             col=col,
                             hue=hue,
                             col_wrap=col_wrap,
                             height=3.0,
                             aspect=1.1,
                             col_order=self.order)

        grid.map(sns.scatterplot, "Width", "Height")
        grid.add_legend()
        if self.out_dir:
            plt.savefig(f'{self.out_dir}/{self.mode}/ScatterPerCat_{self.mode}.png')
        if bool(self.show):
            plt.show()
        plt.close()

    def show_scatterplot_bysize(self):

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1 = sns.scatterplot(data=self.df,
                              x='Width',
                              y='Height',
                              size=self.df['area'],
                              hue=self.df['label_names'],
                              sizes=(20, 600))

        plt.title(f'Scatter plot - {self.mode}')
        if self.out_dir:
            plt.savefig(f'{self.out_dir}/{self.mode}/ScatterPlot_{self.mode}.png', dpi=200)
        if bool(self.show):
            plt.show()
        plt.close()

    def show_kmeans(self, dataframe, hue, centers):

        ax = sns.scatterplot(
            data=dataframe,
            x='Width',
            y='Height',
            hue='cluster',
            palette="deep",
            alpha=0.8,
            s=10)

        ax = sns.scatterplot(
            centers[:, 0],
            centers[:, 1],
            hue=range(hue),
            palette="deep",
            s=20,
            ec='black',
            legend=False,
            ax=ax)

        plt.axis('square')
        plt.grid()
        ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.title(f'K-MEANS_{self.mode}')
        if self.out_dir:
            plt.savefig(f'{self.out_dir}/{self.mode}/Kmeans_{self.mode}.png')
        if bool(self.show):
            plt.show()
        plt.close()

    def show_priorsgtIOU_distribution(self, iou, input_shape):
        fig, ax = plt.subplots(1)
        q25, q75 = np.percentile(iou, [25, 75])
        bin_width = 2 * (q75 - q25) * len(iou) ** (-1 / 3)
        bins = round((iou.max() - iou.min()) / bin_width)
        _, binss, _ = ax.hist(iou, bins=bins, range=[0., 1.], density=True)

        mean, std = np.mean(iou).round(3), np.std(iou).round(3)
        x = np.linspace(min(binss), max(binss), 100)

        # plot norm distribution ###########################################
        pdf = norm.pdf(x, loc=mean, scale=std)
        ax.plot(x, pdf, linestyle='dashed', c='red', lw=1,
                label=f'mean: {mean}\nstd: {std}')

        ax.legend(loc='upper left', frameon=False)
        plt.title(f"{self.mode}: PRIORS-GTs-IOU Distribution @ SCALE_{input_shape[0]}_{input_shape[1]}")
        plt.grid()
        plt.savefig(f'{self.out_dir}/{self.mode}/PriorIOUGt_{self.mode}.png', dpi=200)
        plt.show()
