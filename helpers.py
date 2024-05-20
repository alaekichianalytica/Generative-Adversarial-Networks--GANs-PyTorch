import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch

from IPython.display import display, HTML
from start_tensorboard import start_tensorboard

# Start TensorBoard
tensorboard_url = start_tensorboard()

# Display the link
display(HTML(f'<a href="{tensorboard_url}" target="_blank">Click here to open TensorBoard</a>'))


# Continue with the Logger class and training loop setup
class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):
        if isinstance(d_error, torch.Tensor):
            d_error = d_error.item()
        if isinstance(g_error, torch.Tensor):
            g_error = g_error.item()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar('{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar('{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format == 'NHWC':
            images = images.permute(0, 3, 1, 2)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        self.writer.add_image(img_name, horizontal_grid, step)
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=False):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        if isinstance(d_error, torch.Tensor):
            d_error = d_error.item()
        if isinstance(g_error, torch.Tensor):
            g_error = g_error.item()
        if isinstance(d_pred_real, torch.Tensor):
            d_pred_real = d_pred_real.mean().item()
        if isinstance(d_pred_fake, torch.Tensor):
            d_pred_fake = d_pred_fake.mean().item()

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, num_epochs, n_batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real, d_pred_fake))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(), '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(), '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise