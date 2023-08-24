from abc import ABC, abstractmethod
import numpy as np


class TaskLogger(ABC):
    @abstractmethod
    def log_text(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_image(self, key, image, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        if hasattr(self, 'pl_logger'):
            return self.pl_logger.log_metrics(*args, **kwargs)
        else:
            raise AttributeError('No Pytorch Lightning logger assigned to the object and no log_metrics method.')

    def log_figure(self, fig_name, fig, step):
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = np.reshape(buf, (h, w, 4))

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)


        self.log_image(fig_name, image=buf, step=step)
