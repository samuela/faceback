import random

import matplotlib.pyplot as plt

import torch
# from kindling.nan_police import torch
from torch.autograd import Variable

# This is actually required to use `torch.utils.data`.
# pylint: disable=unused-import
import torchvision

from kindling.distributions import BernoulliNet, Normal, NormalNet, Normal_MeanPrecisionNet
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer

from bars_data import sample_many_one_bar_images
from utils import no_ticks


torch.manual_seed(0)
random.seed(0)

num_samples = 10000
img_size = 4

batch_size = 32
num_groups = 1
dim_z = img_size

data_tensor = sample_many_one_bar_images(num_samples, img_size)
data_tensor += 0.1 * torch.randn(num_samples, img_size, img_size)

train_loader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(data_tensor, torch.zeros(num_samples)),
  batch_size=batch_size,
  shuffle=True
)

encode = torch.nn.Linear(img_size * img_size, dim_z)
decode = torch.nn.Linear(dim_z, img_size * img_size)

optimizer = torch.optim.Adam([
  {'params': encode.parameters(), 'lr': 1e-3},
  {'params': decode.parameters(), 'lr': 1e-3}
])

def train(num_epochs, show_viz=True):
  loss_per_iter = []

  for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
      # The final batch may not have the same size as `batch_size`.
      actual_batch_size = data.size(0)

      data_var = Variable(data.view(actual_batch_size, -1))
      loss = (decode(encode(data_var)) - data_var).pow(2).sum() / actual_batch_size

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_per_iter.append(loss.data[0])
      print(f'Epoch {epoch}, {batch_idx} / {len(train_loader)}')
      print(f'  Loss: {loss.data[0]}')

    if show_viz:
      plt.figure()
      plt.plot(loss_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('mean squared error')

      # This has a good mix when img_size = 4
      viz_reconstruction(123456, epoch)
      plt.show()

def viz_reconstruction(plot_seed, epoch):
  pytorch_rng_state = torch.get_rng_state()
  python_rng_state = random.getstate()

  torch.manual_seed(plot_seed)
  random.seed(plot_seed)

  # grab random sample from train_loader
  train_sample, _ = iter(train_loader).next()
  reconstr = decode(encode(Variable(train_sample.view(batch_size, -1)))).data

  _, ax = plt.subplots(2, 8, figsize=(12, 4))

  for i in range(8):
    ax[0, i].imshow(train_sample[i].numpy(), vmin=0, vmax=1)
    ax[1, i].imshow(reconstr[i].view(4, 4).numpy(), vmin=0, vmax=1)

    no_ticks(ax[0, i])
    no_ticks(ax[1, i])

  ax[0, 0].set_ylabel('true')
  ax[1, 0].set_ylabel('reconst')

  plt.tight_layout()
  plt.suptitle(f'Epoch {epoch}')

  torch.set_rng_state(pytorch_rng_state)
  random.setstate(python_rng_state)

# if __name__ == '__main__':
#   train(10)
