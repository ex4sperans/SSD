import os

from matplotlib import pyplot as plt

from misc import height_and_width


def plot_image(image, save_path=None, file_name=None, dpi=800, save=True):

    height, width = height_and_width(image.shape)

    fig = plt.figure(figsize=(width / dpi, height / dpi), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(image, aspect='auto', interpolation="none")

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, file_name + ".jpg"), dpi=dpi)
        plt.close()
    else:
        return fig, ax


def plot_with_bboxes(image, bboxes, save_path, file_name, dpi=800):

    def add_bboxes(ax, bboxes):

        for classname, bbox in bboxes.iterrows():
            xmin, ymin, xmax, ymax, *centerbox = bbox
            bx = (xmin, xmax, xmax, xmin, xmin)
            by = (ymin, ymin, ymax, ymax, ymin)
            ax.plot(bx, by, c="blue", lw=0.5)
        
            bbox_props = dict(boxstyle="square,pad=0.3",
                              fc="blue", ec="blue", lw=0.5)
            text = classname
            ax.text(xmin, ymin, text, ha='center', va='center',
                    size=1, color='black', bbox=bbox_props,
                    fontweight='semibold')

    fig, ax = plot_image(image, save=False)
    add_bboxes(ax, bboxes)

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, file_name + ".jpg"), dpi=dpi)
    plt.close()
