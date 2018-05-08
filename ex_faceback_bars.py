import itertools
import random

import matplotlib.pyplot as plt

import torch
# from kindling.nan_police import torch
from torch.autograd import Variable

# This is actually required to use `torch.utils.data`.
# pylint: disable=unused-import
import torchvision

from kindling.distributions import (
  BernoulliNet,
  Normal,
  NormalNet,
  Normal_MeanPrecisionNet
)
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer

from bars_data import sample_many_one_bar_images
from faceback import FacebackVAE, FacebackAveragingVAE, VotingMixtureWeightNet, UniformMixtureWeightNet, SparseProductOfExpertsVAE
from utils import no_ticks


torch.manual_seed(0)
random.seed(0)

num_samples = 10000
img_size = 4

batch_size = 32
num_groups = img_size
dim_xs = [img_size] * num_groups
dim_z = 8

data_tensor = sample_many_one_bar_images(num_samples, img_size)
data_tensor += 0.01 * torch.randn(num_samples, img_size, img_size)

train_loader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(data_tensor, torch.zeros(num_samples)),
  batch_size=batch_size,
  shuffle=True
)

total_iterations = 0

def sample_random_mask(rows, cols):
  combos = list(itertools.product([0, 1], repeat=cols))[1:]
  return torch.FloatTensor([random.choice(combos) for _ in range(rows)])

def make_generative_net(dim_x):
  return NormalNet(
    torch.nn.Linear(dim_z, dim_x),
    torch.nn.Sequential(
      torch.nn.Linear(dim_z, dim_x),
      Lambda(lambda x: torch.exp(0.5 * x))
    )
  )

baseline_precision = Variable(100 * torch.ones(1), requires_grad=True)

def make_inference_net(dim_x):
  # return NormalNet(
  #   mu_net=torch.nn.Sequential(
  #     shared,
  #     torch.nn.Linear(dim_shared, dim_z)
  #   ),
  #   sigma_net=torch.nn.Sequential(
  #     shared,
  #     torch.nn.Linear(dim_shared, dim_z),
  #     Lambda(lambda x: torch.exp(0.5 * x))
  #   )
  # )

  return Normal_MeanPrecisionNet(
    torch.nn.Linear(dim_x, dim_z),
    torch.nn.Sequential(
      torch.nn.Linear(dim_x, dim_z),
      Lambda(lambda x: torch.exp(x) * 0 + baseline_precision)
    )
  )

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)),
  Variable(torch.ones(1, dim_z))
)

lam = 1
sgd_lr = 1e-4

vae = SparseProductOfExpertsVAE(
  inference_nets=[make_inference_net(dim_x) for dim_x in dim_xs],
  generative_nets=[make_generative_net(dim_x) for dim_x in dim_xs],
  prior_z=prior_z,
  prior_theta=NormalPriorTheta(sigma=1),
  lam=lam
)

optimizer = MetaOptimizer([
  torch.optim.Adam(
    set(p for net in vae.inference_nets for p in net.parameters()),
    lr=1e-3
  ),
  torch.optim.Adam(
    set(p for net in vae.generative_nets for p in net.parameters()),
    lr=1e-3
  ),
  torch.optim.Adam([baseline_precision], lr=1e-3),
  torch.optim.SGD([vae.sparsity_matrix], lr=sgd_lr)
])

def train(num_epochs, show_viz=True):
  global total_iterations

  elbo_per_iter = []

  for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
      # The final batch may not have the same size as `batch_size`.
      actual_batch_size = data.size(0)

      info = vae.elbo(
        Xs=[Variable(data[:, i]) for i in range(num_groups)],
        group_mask=Variable(torch.ones(actual_batch_size, num_groups)),
        inference_group_mask=Variable(
          # sample_random_mask(actual_batch_size, num_groups)
          torch.ones(actual_batch_size, num_groups)
          # torch.FloatTensor([1, 0, 0, 0]).expand(actual_batch_size, -1)
        )
      )
      elbo = info['elbo']
      # q_z = info['q_z']
      # print(q_z.mu.data)
      # print(q_z.sigma.data)

      optimizer.zero_grad()
      (-elbo).backward()
      optimizer.step()
      vae.proximal_step(sgd_lr * lam)

      elbo_per_iter.append(elbo.data[0])
      total_iterations += 1
      print(f'Epoch {epoch}, {batch_idx} / {len(train_loader)}')
      print(f'  ELBO: {elbo.data[0]}')
      # TODO print ELBO breakdown

    if show_viz:
      viz_elbo(elbo_per_iter)
      viz_sparsity()

      # This has a good mix when img_size = 4
      viz_reconstruction(123456, epoch)

      plt.show()

def viz_elbo(elbo_per_iter):
  plt.figure()
  plt.plot(elbo_per_iter)
  plt.xlabel('iteration')
  plt.ylabel('ELBO')

def viz_sparsity():
  plt.figure()
  plt.imshow(vae.sparsity_matrix.data.numpy())
  plt.colorbar()
  plt.xlabel('latent components')
  plt.ylabel('groups')

def viz_reconstruction(plot_seed, epoch):
  pytorch_rng_state = torch.get_rng_state()
  python_rng_state = random.getstate()

  torch.manual_seed(plot_seed)
  random.seed(plot_seed)

  # grab random sample from train_loader
  train_sample, _ = iter(train_loader).next()
  inference_group_mask = Variable(
    # sample_random_mask(batch_size, num_groups)
    torch.ones(batch_size, num_groups)
    # torch.FloatTensor([1, 0, 0, 0]).expand(batch_size, -1)
  )
  info = vae.reconstruct(
    [Variable(train_sample[:, i]) for i in range(num_groups)],
    inference_group_mask
  )
  reconstr = info['reconstructed']
  reconstr_tensor = torch.stack([reconstr[i].mu.data for i in range(num_groups)], dim=1)
  # mix_samples = info['mixture_sample']

  # train_sample_untransformed = untransform(train_sample)
  # reconstr_untransformed = untransform([d.rate.data for d in reconstr])

  _, ax = plt.subplots(2, 8, figsize=(12, 4))

  for i in range(8):
    ax[0, i].imshow(train_sample[i].numpy(), vmin=0, vmax=1)
    ax[1, i].imshow(reconstr_tensor[i].numpy(), vmin=0, vmax=1)

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
