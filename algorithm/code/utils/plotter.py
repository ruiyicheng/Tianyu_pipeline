import matplotlib.pyplot as plt
import numpy as np

class plotter:
    def __init__(self):
        pass

    def plot_image(self, image, title=None, cmap='gray'):
        m = np.mean(image)
        s = np.std(image)

        plt.imshow(image, cmap=cmap,vmin=m-1*s, vmax=m+1*s)
        plt.title(title)
        plt.colorbar()
        plt.show()
    def plot_image_with_star(self, image, star_pos, title=None, cmap='gray'):
        m = np.mean(image)
        s = np.std(image)

        plt.imshow(image, cmap=cmap,vmin=m-1*s, vmax=m+1*s)
        plt.title(title)
        plt.colorbar()
        plt.scatter(star_pos['x'], star_pos['y'], c='red', s=10)
        plt.show()

