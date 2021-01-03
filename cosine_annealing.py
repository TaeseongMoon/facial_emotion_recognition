from math import floor, pi, cos
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, initial_learning_rate, n_epoch=160, alpha=1e-6, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.alpha = alpha
        self.n_epoch = n_epoch
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
 
        step = min(epoch, self.n_epoch)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.n_epoch))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        lr = self.initial_learning_rate*decayed
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)