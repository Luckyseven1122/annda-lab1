import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Animator:

    EPOCHS_PER_FRAME = 25
    ANIM_LEN = 10

    C_PRED_TRAIN = '#ff5050'
    C_TARGET_TRAIN = '#6699ff'
    C_PRED_VALID = '#990000'
    C_TARGET_VALID = '#000099'

    def __init__(self, t_train, target_train, t_valid, target_valid):
        self.t_train = t_train
        self.t_valid = t_valid
        self.target_train = target_train
        self.target_valid = target_valid
        self.pred_train = []
        self.pred_valid = []
        self.epochs = []

    def add_epoch(self, epoch, pred_train, pred_valid):
        self.epochs.append(epoch)
        self.pred_train.append(pred_train)
        self.pred_valid.append(pred_valid)

    def set_error_curves(self, error_train, error_valid):
        self.error_train = error_train
        self.error_valid = error_valid

    def save_animation(self, filepath):
        self.init_fig()
        fps = len(self.epochs)/self.ANIM_LEN

        an = animation.FuncAnimation(self.fig, self.plot_anim_frame,
                                     interval=1,
                                     frames=len(self.epochs)
                                     )
        ext = filepath.split('.')[-1]
        writer, kwargs = {
            'mp4': (animation.FFMpegWriter(fps=fps), dict()),
            'gif': ('imagemagick', {'fps': fps})
        }.get(ext)
        an.save(filepath, writer=writer, **kwargs)

    def init_fig(self):
        self.fig = plt.figure(figsize=(10, 6), dpi=220)
        self.ax_train = self.fig.add_subplot(221)
        self.ax_valid = self.fig.add_subplot(222)
        self.ax_train_error = self.fig.add_subplot(223)
        self.ax_valid_error = self.fig.add_subplot(224)

        self.ax_train.set_ylim(0.2, 1.4)
        self.ax_valid.set_ylim(0.2, 1.4)
        self.ax_train.set_xlim(self.t_train[0], self.t_train[-1])
        self.ax_valid.set_xlim(self.t_valid[0], self.t_valid[-1])
        self.ax_train.set_title('Training set')
        self.ax_valid.set_title('Validation set')
        self.ax_train.plot(self.t_train, self.target_train, c=self.C_TARGET_TRAIN)
        self.ax_valid.plot(self.t_valid, self.target_valid, c=self.C_TARGET_VALID)

        self.ax_train_error.set_xlim(0, self.epochs[-1])
        self.ax_valid_error.set_xlim(0, self.epochs[-1])
        self.ax_train_error.plot(self.error_train, c='black')
        self.ax_valid_error.plot(self.error_valid, c='black')

    def plot_anim_frame(self, frame):
        try:
            for a in [
                self.ax_train,
                self.ax_valid,
                self.ax_train_error,
                self.ax_valid_error
            ]:
                a.lines.pop(1)
        except IndexError:
            pass

        self.ax_train.plot(self.t_train, self.pred_train[frame], c=self.C_PRED_TRAIN)
        self.ax_valid.plot(self.t_valid, self.pred_valid[frame], c=self.C_PRED_VALID)

        curr_epoch_line_x = (self.epochs[frame], self.epochs[frame])
        curr_epoch_line_y = (0, 1)
        self.ax_train_error.plot(curr_epoch_line_x, curr_epoch_line_y, c='black')
        self.ax_valid_error.plot(curr_epoch_line_x, curr_epoch_line_y, c='black')
        self.fig.suptitle('Epoch {}'.format(self.epochs[frame]))
