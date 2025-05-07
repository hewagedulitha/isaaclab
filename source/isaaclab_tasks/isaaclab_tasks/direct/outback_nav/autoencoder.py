import math
import copy
import warnings
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Sequence, Union, Tuple

def fit_ae(model, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., device="cuda", **kwargs):
    """
    Training functions for the AEs
    :param model: model to train
    :param mode: (str) {'basic | 'contractive' | 'denoising'}
    :param tr_data: (optional) specific training data to use
    :param val_data: (optional) specific validation data to use
    :param num_epochs: (int) number of epochs
    :param bs: (int) batch size
    :param lr: (float) learning rate
    :param momentum: (float) momentum coefficient
    :return: history of training (like in Keras)
    """
    mode_values = (None, 'basic', 'contractive', 'denoising')
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1 and mode in mode_values

    # set the device: GPU if cuda is available, else CPU
    model.to(device)

    # set optimizer, loss type and datasets (depending on the type of AE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    if mode == 'denoising':
        if tr_data is not None or val_data is not None:
            warnings.warn("'denoising' flag was set, so NoisyMNIST will be used for training and validation")
        noisy_train, noisy_val = get_noisy_sets(**kwargs)
        tr_data, tr_targets = noisy_train.data, noisy_train.targets
        val_data, val_targets = noisy_val.data, noisy_val.targets
        del noisy_train, noisy_val
    else:
        tr_set, val_set = get_clean_sets()
        if tr_data is None:
            tr_data, tr_targets = tr_set.data, tr_set.targets
        else:
            tr_data = tr_data.to(device)
            tr_targets = torch.flatten(copy.deepcopy(tr_data), start_dim=1)
        if val_data is None:
            val_data, val_targets = val_set.data, val_set.targets
        else:
            val_data = val_data.to(device)
            val_targets = torch.flatten(copy.deepcopy(val_data), start_dim=1)
        del tr_set, val_set
    if 'ConvAutoencoder' in model.__class__.__name__:
        val_bs = bs
        tr_data, tr_targets = tr_data.cpu(), tr_targets.cpu()
        val_data, val_targets = val_data.cpu(), val_targets.cpu()
    else:
        val_bs = None
    torch.cuda.empty_cache()

    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0
        n_batches = math.ceil(len(tr_data) / bs)
        # shuffle
        indexes = torch.randperm(tr_data.shape[0])
        tr_data = tr_data[indexes]
        tr_targets = tr_targets[indexes]
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # zero the gradient
            optimizer.zero_grad()
            # select a (mini)batch from the training set and compute net's outputs
            train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to(device)
            train_targets_batch = tr_targets[batch_idx * bs: batch_idx * bs + bs].to(device)
            outputs = model(train_data_batch)
            # compute loss (flatten output in case of ConvAE. targets already flat)
            loss = criterion(torch.flatten(outputs, 1), train_targets_batch)
            tr_loss += loss.item()
            # propagate back the loss
            loss.backward()
            optimizer.step()
            # update progress bar
            progbar.update()
            progbar.set_postfix(train_loss=f"{loss.item():.4f}")
        last_batch_loss = loss.item()
        tr_loss /= n_batches
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate(model=model, data=val_data, targets=val_targets, criterion=criterion, bs=val_bs)
        history['val_loss'].append(round(val_loss, 5))
        torch.cuda.empty_cache()
        progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{val_loss:.4f}")
        progbar.close()

        # simple early stopping mechanism
        if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (last_values[-3] < last_values[-2] < last_values[-1]):
                return history

    return history


def evaluate(model, criterion, mode='basic', data=None, targets=None, bs=None, **kwargs):
    """ Evaluate the model """
    # set the data
    if data is None:
        _, val_set = get_noisy_sets(**kwargs) if mode == 'denoising' else get_clean_sets()
        data, targets = val_set.data, val_set.targets
    bs = len(data) if bs is None else bs
    n_batches = math.ceil(len(data) / bs)
    if 'ConvAutoencoder' in model.__class__.__name__:
        data = data.to('cpu')
        targets = targets.to('cpu')
    else:
        data = data.to(device)
        targets = targets.to(device)

    # evaluate
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx in range(n_batches):
            data_batch = data[batch_idx * bs: batch_idx * bs + bs].to(device)
            targets_batch = targets[batch_idx * bs: batch_idx * bs + bs].to(device)
            outputs = model(data_batch)
            # flatten outputs in case of ConvAE (targets already flat)
            loss = criterion(torch.flatten(outputs, 1), targets_batch)
            val_loss += loss.item()
    return val_loss / n_batches

class AbstractAutoencoder(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)

    def show_manifold_convergence(self, load=None, path=None, max_iters=1000, thresh=0.02, side_len=28, save=False):
        """
        Show the manifold convergence of an AE when fed with random noise.
        The output of the AE is fed again as input in an iterative process.
        :param load: if True, load an images progression of the manifold convergence
        :param path: path of the images progression
        :param max_iters: max number of iterations.
        :param thresh: threshold of MSE between 2 iterations under which the process is stopped
        :param side_len: length of the side of the images
        :param save: if True, save the images progression and the animation
        """
        if load:
            images_progression = np.load(path)
        else:
            self.cpu()
            noise_img = torch.randn((1, 1, side_len, side_len))
            noise_img -= torch.min(noise_img)
            noise_img /= torch.max(noise_img)
            images_progression = [torch.squeeze(noise_img)]
            serializable_progression = [torch.squeeze(noise_img).cpu().numpy()]

            # iterate
            i = 0
            loss = 1000
            input = noise_img
            prev_output = None
            with torch.no_grad():
                while loss > thresh and i < max_iters:
                    output = self(input)
                    img = torch.reshape(torch.squeeze(output), shape=(side_len, side_len))
                    rescaled_img = (img - torch.min(img)) / torch.max(img)
                    images_progression.append(rescaled_img)
                    serializable_progression.append(rescaled_img.cpu().numpy())
                    if prev_output is not None:
                        loss = F.mse_loss(output, prev_output)
                    prev_output = output
                    input = output
                    i += 1

            # save sequence of images
            if save:
                serializable_progression = np.array(serializable_progression)
                np.save(file="manifold_img_seq", arr=serializable_progression)

        if save:
            images_progression = images_progression[:60]
            frames = []  # for storing the generated images
            fig = plt.figure()
            for i in range(len(images_progression)):
                frames.append([plt.imshow(images_progression[i], animated=True)])
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            ani.save('movie.gif')
            plt.show()
        else:
            # show images progression
            img = None
            for i in range(len(images_progression)):
                if img is None:
                    img = plt.imshow(images_progression[0])
                else:
                    img.set_data(images_progression[i])
                plt.pause(.1)
                plt.draw()

class DeepConvAutoencoder(AbstractAutoencoder):
    """ Conv Ae with variable number of conv layers """
    def __init__(self, inp_side_len=28, dims: Sequence[int] = (5, 10),
                 kernel_sizes: int = 3, central_dim=100, pool=True):
        super().__init__()
        self.type = "deepConvAE"

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        side_len = inp_side_len
        side_lengths = [side_len]
        dims = (1, *dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if pool and (i % step_pool == 0 or i == len(dims) - 1) and side_len > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                side_len = math.floor(side_len / 2)
                side_lengths.append(side_len)

        # fully connected layers in the center of the autoencoder to reduce dimensionality
        fc_dims = (side_len ** 2 * dims[-1], side_len ** 2 * dims[-1] // 2, central_dim)
        self.encoder = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(inplace=True)
        )

        # build decoder
        central_side_len = side_lengths.pop(-1)
        # side_lengths = side_lengths[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(side_lengths) > 0 and side_len * 2 == side_lengths.pop(-1) else 3
            pad, stride = (1, 1) if side_len == inp_side_len else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(fc_dims[2], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[-1], central_side_len, central_side_len)),
            *dec_layers,
        )
