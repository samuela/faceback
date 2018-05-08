import itertools
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
# from kindling.nan_police import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from kindling.distributions import BernoulliNet, Normal, Normal_MeanPrecisionNet
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer

from faceback import SparseProductOfExpertsVAE


torch.manual_seed(0)
random.seed(0)

batch_size = 128
num_groups = 4
dim_shared = 256
dim_xs = [784 // num_groups] * num_groups
dim_z = 20

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda x: [
    x[:, :14, :14].contiguous().view(-1),
    x[:, :14, 14:].contiguous().view(-1),
    x[:, 14:, :14].contiguous().view(-1),
    x[:, 14:, 14:].contiguous().view(-1)
  ])
])

def untransform(Xs):
  reshaped = [x.view(-1, 14, 14) for x in Xs]
  return torch.cat([
    torch.cat([reshaped[0], reshaped[1]], dim=2),
    torch.cat([reshaped[2], reshaped[3]], dim=2),
  ], dim=1)

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data/mnist/', train=True, download=True, transform=transform),
  batch_size=batch_size,
  shuffle=True
)
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data/mnist/', train=False, transform=transform),
  batch_size=batch_size,
  shuffle=True
)

def sample_random_mask(rows, cols):
  combos = list(itertools.product([0, 1], repeat=cols))[1:]
  return torch.FloatTensor([random.choice(combos) for _ in range(rows)])

def make_generative_net(dim_x):
  return BernoulliNet(
    torch.nn.Sequential(
      torch.nn.Linear(dim_z, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, dim_x),
      torch.nn.Sigmoid()
    )
  )

def make_shared_inference_net(dim_x):
  return torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_shared),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_shared, dim_shared),
    torch.nn.ReLU()
  )

shared_inference_nets = [make_shared_inference_net(dim_x) for dim_x in dim_xs]

def make_inference_net(shared):
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
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(dim_shared, dim_z)
    ),
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(dim_shared, dim_z),
      Lambda(lambda x: torch.exp(0.5 * x))
    )
  )

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)),
  Variable(torch.ones(1, dim_z))
)

# vae = FacebackAveragingVAE(
#   inference_nets=[make_inference_net(sh) for sh in shared_inference_nets],
#   generative_nets=[make_generative_net(dim_x) for dim_x in dim_xs],
#   prior_z=prior_z
# )

# vae = FacebackVAE(
#   inference_nets=[make_inference_net(sh) for sh in shared_inference_nets],
#   generative_nets=[make_generative_net(dim_x) for dim_x in dim_xs],
#   # mixture_weight_net=VotingMixtureWeightNet(
#   #   embeddings=shared_inference_nets,
#   #   W=Variable(torch.randn(num_groups, dim_shared), requires_grad=True),
#   #   b=Variable(torch.randn(num_groups), requires_grad=True)
#   # ),
#   mixture_weight_net=UniformMixtureWeightNet(),
#   prior_z=prior_z
# )

lam = 0.0
sgd_lr = 1e-5

vae = SparseProductOfExpertsVAE(
  inference_nets=[make_inference_net(sh) for sh in shared_inference_nets],
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
  torch.optim.SGD([vae.sparsity_matrix], lr=sgd_lr)
])

def train(num_epochs, show_viz=True):
  elbo_per_iter = []

  for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
      # The final batch may not have the same size as `batch_size`.
      actual_batch_size = data[0].size(0)

      info = vae.elbo(
        Xs=[Variable(x) for x in data],
        group_mask=Variable(torch.ones(actual_batch_size, num_groups)),
        inference_group_mask=Variable(
          # sample_random_mask(actual_batch_size, num_groups)
          torch.ones(actual_batch_size, num_groups)
          # torch.FloatTensor([1, 0, 0, 0]).expand(actual_batch_size, -1)
        )
      )
      elbo = info['elbo']

      optimizer.zero_grad()
      (-elbo).backward()
      optimizer.step()
      vae.proximal_step(sgd_lr * lam)

      # TODO figure out the right way to evaluate solution collapses.
      # if torch.max(torch.abs(reconstr.rate.data[0] - reconstr.rate.data)) < 1e-4:
      #   print('ya done collapsed son. try reducing your learning rate.')

      elbo_per_iter.append(elbo.data[0])
      print(f'Epoch {epoch}, {batch_idx} / {len(train_loader)}')
      print(f'  ELBO: {elbo.data[0]}')

    if show_viz:
      plt.figure()
      plt.plot(elbo_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('ELBO')

      viz_reconstruction(0, epoch)
      plt.show()

def no_ticks(ax):
  ax.tick_params(
    axis='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labeltop=False,
    labelleft=False,
    labelright=False
  )

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
    [Variable(x) for x in train_sample],
    inference_group_mask
  )
  reconstr = info['reconstructed']
  # mix_samples = info['mixture_sample']

  train_sample_untransformed = untransform(train_sample)
  reconstr_untransformed = untransform([d.rate.data for d in reconstr])

  _, ax = plt.subplots(4, 8, figsize=(12, 4))

  for i in range(8):
    ax[0, i].imshow(train_sample_untransformed[i].numpy())

    mask1 = np.kron(
      np.maximum(inference_group_mask[i, :].view(2, 2).data.numpy(), 0.1),
      np.ones((14, 14))
    )
    ax[1, i].imshow(train_sample_untransformed[i].numpy() * mask1)

    # mask2 = np.kron(
    #   np.maximum(np.eye(4)[mix_samples.data[i, 0]].reshape(2, 2), 0.1),
    #   np.ones((14, 14))
    # )
    # ax[2, i].imshow(train_sample_untransformed[i].numpy() * mask2)

    ax[3, i].imshow(reconstr_untransformed[i].numpy())

    no_ticks(ax[0, i])
    no_ticks(ax[1, i])
    no_ticks(ax[2, i])
    no_ticks(ax[3, i])

  ax[0, 0].set_ylabel('true')
  ax[1, 0].set_ylabel('observed')
  ax[2, 0].set_ylabel('selected')
  ax[3, 0].set_ylabel('reconst')

  plt.tight_layout()
  plt.suptitle(f'Epoch {epoch}')

  torch.set_rng_state(pytorch_rng_state)
  random.setstate(python_rng_state)

if __name__ == '__main__':
  train(10)
