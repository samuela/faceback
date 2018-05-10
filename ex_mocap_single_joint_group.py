"""Running faceback on the mocap skeleton data with each joint corresponding to
a single group."""

import itertools
import numpy as np

# For some reason this is required even though it's never even used.
# pylint: disable=unused-import
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# This is actually required to use `torch.utils.data`.
# pylint: disable=unused-import
import torchvision
import torch
from torch.autograd import Variable

from faceback import SparseProductOfExpertsVAE

from kindling.distributions import (
  BernoulliNet,
  Normal,
  NormalNet,
  Normal_MeanPrecisionNet
)
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer

from mocap import mocap_data

from utils import sample_random_mask

subject = 7
train_trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_trials = [11] #[11, 12]
skeleton = mocap_data.load_skeleton(subject)

dim_z = 32
batch_size = 64
lam = 0
sparsity_matrix_lr = 1e-3

joint_order = [
  'root',
  'lowerback',
  'upperback',
  'thorax',
  'lowerneck',
  'upperneck',
  'head',
  'rclavicle',
  'rhumerus',
  'rradius',
  'rwrist',
  'rhand',
  'rfingers',
  'rthumb',
  'lclavicle',
  'lhumerus',
  'lradius',
  'lwrist',
  'lhand',
  'lfingers',
  'lthumb',
  'rfemur',
  'rtibia',
  'rfoot',
  'rtoes',
  'lfemur',
  'ltibia',
  'lfoot',
  'ltoes'
]
num_groups = len(joint_order)

train_trials_data = [
  mocap_data.load_trial(subject, trial, joint_order=joint_order)
  for trial in train_trials
]
test_trials_data = [
  mocap_data.load_trial(subject, trial, joint_order=joint_order)
  for trial in test_trials
]
_, joint_dims, _ = train_trials_data[0]

# We remove the first three components since those correspond to root position
# in 3d space.
joint_dims['root'] = joint_dims['root'] - 3
Xtrain_raw = torch.FloatTensor(
  # Chain all of the different lists together across the trials
  list(itertools.chain(*[arr for _, _, arr in train_trials_data]))
)[:, 3:]
Xtest_raw = torch.FloatTensor(
  # Chain all of the different lists together across the trials
  list(itertools.chain(*[arr for _, _, arr in test_trials_data]))
)[:, 3:]

# Normalize each of the channels to be within [0, 1].
mins, _ = torch.min(Xtrain_raw, dim=0)
maxs, _ = torch.max(Xtrain_raw, dim=0)

def preprocess(x):
  # Some of these things aren't used, and we don't want to divide by zero
  return (x - mins) / torch.clamp(maxs - mins, min=0.1)

def preprocess_inverse(y):
  return y * torch.clamp(maxs - mins, min=0.1) + mins

Xtrain = preprocess(Xtrain_raw)
Xtest = preprocess(Xtest_raw)

train_loader = torch.utils.data.DataLoader(
  # TensorDataset is stupid. We have to provide two tensors.
  torch.utils.data.TensorDataset(Xtrain, torch.zeros(Xtrain.size(0))),
  batch_size=batch_size,
  shuffle=True
)

def make_generative_net(dim_x):
  hidden_size = 32
  shared = torch.nn.Sequential(
    torch.nn.Linear(dim_z, hidden_size),
    torch.nn.ReLU()
  )
  return NormalNet(
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(hidden_size, dim_x)
    ),
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(hidden_size, dim_x),
      # Learn the log variance
      Lambda(lambda x: torch.exp(0.5 * x))
    )
  )

baseline_precision = Variable(100 * torch.ones(1), requires_grad=True)
def make_inference_net(dim_x):
  hidden_size = 32
  shared = torch.nn.Sequential(
    torch.nn.Linear(dim_x, hidden_size),
    torch.nn.ReLU()
  )
  return Normal_MeanPrecisionNet(
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(hidden_size, dim_z)
    ),
    torch.nn.Sequential(
      shared,
      torch.nn.Linear(hidden_size, dim_z),
      Lambda(lambda x: torch.exp(x) + baseline_precision)
    )
  )

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)),
  Variable(torch.ones(1, dim_z))
)

vae = SparseProductOfExpertsVAE(
  inference_nets=[make_inference_net(joint_dims[j]) for j in joint_order],
  generative_nets=[make_generative_net(joint_dims[j]) for j in joint_order],
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
  torch.optim.SGD([vae.sparsity_matrix], lr=sparsity_matrix_lr)
])

def split_into_groups(data):
  poop = np.cumsum([0] + [joint_dims[j] for j in joint_order])
  return [data[:, poop[i]:poop[i + 1]] for i in range(num_groups)]

def test_loglik():
  Xs = [Variable(x) for x in split_into_groups(Xtest)]
  group_mask = Variable(torch.ones(Xtest.size(0), num_groups))
  q_z = vae.approx_posterior(Xs, group_mask)
  return vae.log_likelihood(Xs, group_mask, q_z.sample())

def train(num_epochs, show_viz=True):
  elbo_per_iter = []
  test_loglik_per_iter = []
  for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
      # The final batch may not have the same size as `batch_size`.
      actual_batch_size = data.size(0)

      info = vae.elbo(
        Xs=[Variable(x) for x in split_into_groups(data)],
        group_mask=Variable(torch.ones(actual_batch_size, num_groups)),
        inference_group_mask=Variable(
          sample_random_mask(actual_batch_size, num_groups)
          # torch.ones(actual_batch_size, num_groups)
        )
      )
      elbo = info['elbo']
      z_kl = info['z_kl']
      reconstruction_log_likelihood = info['reconstruction_log_likelihood']
      logprob_theta = info['logprob_theta']
      logprob_L1 = info['logprob_L1']
      test_ll = test_loglik().data[0]

      optimizer.zero_grad()
      (-elbo).backward()
      optimizer.step()
      vae.proximal_step(sparsity_matrix_lr * lam)

      elbo_per_iter.append(elbo.data[0])
      test_loglik_per_iter.append(test_ll)
      print(f'Epoch {epoch}, {batch_idx} / {len(train_loader)}')
      print(f'  ELBO: {elbo.data[0]}')
      print(f'    -KL(q(z) || p(z)): {-z_kl.data[0]}')
      print(f'    loglik_term      : {reconstruction_log_likelihood.data[0]}')
      print(f'    log p(theta)     : {logprob_theta.data[0]}')
      print(f'    L1               : {logprob_L1.data[0]}')
      print(f'  test log lik.      : {test_ll}')

    if show_viz:
      # ELBO per iteration
      plt.figure()
      plt.plot(elbo_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('ELBO')

      # sparsity matrix
      # _, ax = plt.subplots()
      # plt.imshow(vae.sparsity_matrix.data.numpy())
      # plt.colorbar()
      # ax.set_xlabel('latent components')
      # ax.set_ylabel('groups')
      # ax.xaxis.tick_top()
      # ax.xaxis.set_label_position('top')
      # ax.xaxis.set_ticks(np.arange(dim_z))
      # ax.xaxis.set_ticklabels(1 + np.arange(dim_z))
      # ax.yaxis.set_ticks(np.arange(len(joint_order)))
      # ax.yaxis.set_ticklabels(joint_order)

      viz_reconstruction(0, epoch)

      plt.show()

def repad_position(tensor):
  """Add the three root position dofs back on to a tensor of angles."""
  return torch.cat([torch.zeros(tensor.size(0), 3)] + [tensor], dim=1)

def viz_reconstruction(plot_seed, epoch):
  num_examples = 6

  pytorch_rng_state = torch.get_rng_state()
  torch.manual_seed(plot_seed)

  # grab random sample from train_loader
  train_sample, _ = iter(train_loader).next()
  inference_group_mask = Variable(
    # sample_random_mask(batch_size, num_groups)
    torch.ones(batch_size, num_groups)
  )
  info = vae.reconstruct(
    [Variable(x) for x in split_into_groups(train_sample)],
    inference_group_mask
  )
  reconstr = info['reconstructed']

  true_angles = split_into_groups(preprocess_inverse(train_sample))

  # Take the mean of p(x | z)
  reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
  reconstr_angles = split_into_groups(preprocess_inverse(reconstr_tensor))

  def frame(ix, angles):
    stuff = {j: list(x[ix].numpy()) for j, x in zip(joint_order, angles)}
    # We can't forget the (unlearned) translation dofs
    stuff['root'] = [0, 0, 0] + stuff['root']
    return stuff

  fig = plt.figure(figsize=(12, 4))
  for i in range(num_examples):
    ax1 = fig.add_subplot(2, num_examples, i + 1, projection='3d')
    ax2 = fig.add_subplot(2, num_examples, i + num_examples + 1, projection='3d')
    mocap_data.plot_skeleton(
      skeleton,
      mocap_data.frame_to_xyz(skeleton, frame(i, true_angles)),
      axes=ax1
    )
    mocap_data.plot_skeleton(
      skeleton,
      mocap_data.frame_to_xyz(skeleton, frame(i, reconstr_angles)),
      axes=ax2
    )

  plt.tight_layout()
  plt.suptitle(f'Epoch {epoch}')

  torch.set_rng_state(pytorch_rng_state)
