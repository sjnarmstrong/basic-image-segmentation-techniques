from os import makedirs
from os.path import dirname
from matplotlib import pyplot as plt
import cv2


class Plotter:
    figure_dict = {}
    plt.interactive(False)

    @staticmethod
    def get_zero_padded_fig_and_ax(figure_name, figsize=[6, 6]):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        Plotter.figure_dict[figure_name] = fig
        return fig, ax

    @staticmethod
    def save_plots(should_save, save_dir, save_format=".png"):
        if not should_save:
            Plotter.figure_dict = {}
            return
        makedirs(save_dir, exist_ok=True)
        for img_name in Plotter.figure_dict:
            Plotter.figure_dict[img_name].savefig(
                save_dir + img_name + save_format, dpi=500, bbox_inches="tight", pad_inches=0)
        Plotter.figure_dict = {}

    @staticmethod
    def no_scale_save_plots(should_save, save_dir, img):
        if not should_save:
            return
        makedirs(dirname(save_dir), exist_ok=True)
        cv2.imwrite(save_dir, img)

    @staticmethod
    def show_plots(should_show):
        if should_show:
            plt.show()
        else:
            plt.close('all')

    @staticmethod
    def create_histogram(data, figure_name, x_label="Intensity level", y_label="Probability",
                         title="Histogram of intensity levels"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(data.flatten(), [i for i in range(257)], normed=True, facecolor='k')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        #plt.axis([40, 160, 0, 0.03])
        plt.grid(True)

        Plotter.figure_dict[figure_name] = fig
