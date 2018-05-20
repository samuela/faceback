import itertools
import json
import multiprocessing
import random
import string
import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
# This is actually required to use `torch.utils.data`.
# pylint: disable=unused-import
import torchvision
# For some reason this is required even though it's never even used.
# pylint: disable=unused-import
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable

from faceback import JMVAEPlus, UniformMixtureWeightNet
from kindling.distributions import Normal, NormalNet
from kindling.utils import Lambda, MetaOptimizer, NormalPriorTheta
from mocap import mocap_data
from utils import sample_random_mask


groups = [
  [
    'root',
    'lowerback',
    'upperback',
    'thorax',
    'lowerneck',
    'upperneck',
    'head'
  ],
  [
    'rclavicle',
    'rhumerus',
    'rradius',
    'rwrist',
    'rhand',
    'rfingers',
    'rthumb'
  ],
  [
    'lclavicle',
    'lhumerus',
    'lradius',
    'lwrist',
    'lhand',
    'lfingers',
    'lthumb',
  ],
  [
    'rfemur',
    'rtibia',
    'rfoot',
    'rtoes',
  ],
  [
    'lfemur',
    'ltibia',
    'lfoot',
    'ltoes'
  ]
]
joint_order = [joint for grp in groups for joint in grp]
num_groups = len(groups)
group_names = [
  'core',
  'right arm',
  'left arm',
  'right leg',
  'left leg'
]

class MocapGroupedJMVAE(object):
  """Experiment with mocap data running with joints grouped together into
  * core and head
  * right arm
  * left arm
  * right leg
  * left leg
  """

  PARAMS = [
    'subject',
    'train_trials',
    'test_trials',
    'dim_z',
    'batch_size',
  ]

  def __init__(
      self,
      subject,
      train_trials,
      test_trials,
      dim_z,
      batch_size,
      base_results_dir=None,
      prefix='mocap_jmvae_'
  ):
    self.subject = subject
    self.train_trials = train_trials
    self.test_trials = test_trials
    self.dim_z = dim_z
    self.batch_size = batch_size
    self.base_results_dir = base_results_dir
    self.prefix = prefix

    self.epoch_counter = itertools.count()
    self.epoch = None
    self.elbo_per_iter = []
    self.test_loglik_per_iter = []

    self.load_data()

    self.vae = JMVAEPlus(
      inference_nets=[self.make_inference_net(sum(self.joint_dims[j] for j in g)) for g in groups],
      generative_nets=[self.make_generative_net(sum(self.joint_dims[j] for j in g)) for g in groups],
      mixture_weight_net=UniformMixtureWeightNet(),
      prior_z=Normal(
        Variable(torch.zeros(1, dim_z)),
        Variable(torch.ones(1, dim_z))
      )
    )

    self.optimizer = torch.optim.Adam(set(self.vae.parameters()), lr=1e-3)

    if self.base_results_dir is not None:
      # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
      self.results_folder_name = self.prefix + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
      self.results_dir = self.base_results_dir / self.results_folder_name
      self._init_results_dir()

  def train(self, num_epochs):
    for self.epoch in itertools.islice(self.epoch_counter, num_epochs):
      for batch_idx, (data, _) in enumerate(self.train_loader):
        # The final batch may not have the same size as `batch_size`.
        actual_batch_size = data.size(0)

        info = self.vae.elbo(
          Xs=[Variable(x) for x in self.split_into_groups(data)],
          group_mask=Variable(torch.ones(actual_batch_size, num_groups)),
          inference_group_mask=Variable(torch.ones(actual_batch_size, num_groups))
        )
        elbo = info['elbo']
        # test_ll = self.test_loglik().data[0]

        self.optimizer.zero_grad()
        (-elbo).backward()
        self.optimizer.step()

        self.elbo_per_iter.append(elbo.data[0])
        # self.test_loglik_per_iter.append(test_ll)
        print(f'Epoch {self.epoch}, {batch_idx} / {len(self.train_loader)}')
        print(f'  ELBO: {elbo.data[0]}')
        # print(f'  test log lik.      : {test_ll}', flush=True)
        # print(self.test_euclidean_error())

      # Checkpoint every once in a while
      if self.epoch % 50 == 0:
        self.checkpoint()

    # Checkpoint at the very end as well
    self.checkpoint()

  def make_generative_net(self, dim_x):
    hidden_size = 16
    shared = torch.nn.Sequential(
      torch.nn.Linear(self.dim_z, hidden_size),
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

  def make_inference_net(self, dim_x):
    hidden_size = 16
    shared = torch.nn.Sequential(
      torch.nn.Linear(dim_x, hidden_size),
      torch.nn.ReLU()
    )
    return NormalNet(
      torch.nn.Sequential(
        shared,
        torch.nn.Linear(hidden_size, self.dim_z)
      ),
      torch.nn.Sequential(
        shared,
        torch.nn.Linear(hidden_size, self.dim_z),
        # Learn the log variance
        Lambda(lambda x: torch.exp(0.5 * x))
      )
    )

  def load_data(self):
    self.skeleton = mocap_data.load_skeleton(self.subject)
    train_trials_data = [
      mocap_data.load_trial(self.subject, trial, joint_order=joint_order)
      for trial in self.train_trials
    ]
    test_trials_data = [
      mocap_data.load_trial(self.subject, trial, joint_order=joint_order)
      for trial in self.test_trials
    ]
    _, self.joint_dims, _ = train_trials_data[0]

    # We remove the first three components since those correspond to root
    # position in 3d space.
    self.joint_dims['root'] = self.joint_dims['root'] - 3

    Xtrain_raw = torch.FloatTensor(
      # Chain all of the different lists together across the trials
      list(itertools.chain(*[arr for _, _, arr in train_trials_data]))
    )[:, 3:]
    Xtest_raw = torch.FloatTensor(
      # Chain all of the different lists together across the trials
      list(itertools.chain(*[arr for _, _, arr in test_trials_data]))
    )[:, 3:]

    # Normalize each of the channels to be within [0, 1].
    self.angular_mins, _ = torch.min(Xtrain_raw, dim=0)
    self.angular_maxs, _ = torch.max(Xtrain_raw, dim=0)

    self.Xtrain = self.preprocess(Xtrain_raw)
    self.Xtest = self.preprocess(Xtest_raw)

    self.train_loader = torch.utils.data.DataLoader(
      # TensorDataset is stupid. We have to provide two tensors.
      torch.utils.data.TensorDataset(self.Xtrain, torch.zeros(self.Xtrain.size(0))),
      batch_size=self.batch_size,
      shuffle=True
    )

  def test_loglik(self):
    num_test_frames = self.Xtest.size(0)
    Xs = [Variable(x) for x in self.split_into_groups(self.Xtest)]
    group_mask = Variable(torch.ones(num_test_frames, num_groups))
    reconstr_likelihoods = self.vae.reconstruct(Xs, group_mask)['reconstructed']

    # This particular bit assumes that all of the groups are always available
    # which is the case with mocap, but you probably shouldn't copy paste this
    # code without thinking.
    return sum(lik.logprob(X) for lik, X in zip(reconstr_likelihoods, Xs)) / num_test_frames

  def test_euclidean_error(self):
    num_test_frames = self.Xtest.size(0)
    true_angles = self.split_into_joints(self.preprocess_inverse(self.Xtest))

    Xs = [Variable(x) for x in self.split_into_groups(self.Xtest)]
    group_mask = Variable(torch.ones(num_test_frames, num_groups))

    reconstr = self.vae.reconstruct(Xs, group_mask)['reconstructed']
    reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
    reconstr_angles = self.split_into_joints(self.preprocess_inverse(reconstr_tensor))

    def frame(ix, angles):
      stuff = {j: list(x[ix].numpy()) for j, x in zip(joint_order, angles)}
      # We can't forget the (unlearned) translation dofs
      stuff['root'] = [0, 0, 0] + stuff['root']
      return stuff

    def error(i):
      true_xyz = mocap_data.frame_to_xyz(self.skeleton, frame(i, true_angles))
      reconst_xyz = mocap_data.frame_to_xyz(self.skeleton, frame(i, reconstr_angles))
      return np.sqrt(np.sum([np.sum(np.power(true_xyz[j] - reconst_xyz[j], 2)) for j in joint_order]))

    return sum(error(i) for i in range(num_test_frames)) / num_test_frames

  def test_angle_error(self):
    num_test_frames = self.Xtest.size(0)
    true_angles = self.preprocess_inverse(self.Xtest)

    Xs = [Variable(x) for x in self.split_into_groups(self.Xtest)]
    group_mask = Variable(torch.ones(num_test_frames, num_groups))

    reconstr = self.vae.reconstruct(Xs, group_mask)['reconstructed']
    reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
    reconstr_angles = self.preprocess_inverse(reconstr_tensor)

    return torch.sum(torch.sqrt(torch.sum((true_angles - reconstr_angles).pow(2), dim=1))) / num_test_frames

  def preprocess(self, x):
    """Preprocess the angular data to lie between 0 and 1."""
    # Some of these things aren't used, and we don't want to divide by zero
    return (x - self.angular_mins) / torch.clamp(self.angular_maxs - self.angular_mins, min=0.1)

  def preprocess_inverse(self, y):
    """Inverse of `preprocess`."""
    return y * torch.clamp(self.angular_maxs - self.angular_mins, min=0.1) + self.angular_mins

  def split_into_groups(self, data):
    group_dims = [sum(self.joint_dims[j] for j in grp) for grp in groups]
    poop = np.cumsum([0] + group_dims)
    return [data[:, poop[i]:poop[i + 1]] for i in range(num_groups)]

  def split_into_joints(self, data):
    poop = np.cumsum([0] + [self.joint_dims[j] for j in joint_order])
    return [data[:, poop[i]:poop[i + 1]] for i in range(len(joint_order))]

  def _init_results_dir(self):
    self.results_dir_params = self.results_dir / 'params.json'
    self.results_dir_elbo = self.results_dir / 'elbo_plot'
    self.results_dir_reconstructions = self.results_dir / 'reconstructions'
    self.results_dir_pickles = self.results_dir / 'pickles'

    # The results_dir should be unique
    self.results_dir.mkdir(exist_ok=False)
    self.results_dir_elbo.mkdir(exist_ok=False)
    self.results_dir_reconstructions.mkdir(exist_ok=False)
    self.results_dir_pickles.mkdir(exist_ok=False)
    json.dump(
      {p: getattr(self, p) for p in MocapGroupedJMVAE.PARAMS},
      open(self.results_dir_params, 'w'),
      sort_keys=True,
      indent=2,
      separators=(',', ': ')
    )

  def checkpoint(self):
    if self.base_results_dir is not None:
      fig = viz_reconstruction(self, 0, num_examples=6)
      plt.savefig(self.results_dir_reconstructions / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig = viz_elbo(self)
      plt.savefig(self.results_dir_elbo / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      dill.dump(self, open(self.results_dir_pickles / f'epoch{self.epoch}.p', 'wb'))
    else:
      viz_reconstruction(self, 0, num_examples=6)
      viz_elbo(self)
      plt.show()

def viz_elbo(experiment):
  """ELBO per iteration"""
  fig = plt.figure()
  plt.plot(experiment.elbo_per_iter)
  plt.xlabel('iteration')
  plt.ylabel('ELBO')
  return fig

def viz_reconstruction(experiment, plot_seed, num_examples):
  pytorch_rng_state = torch.get_rng_state()
  torch.manual_seed(plot_seed)

  # grab random sample from train_loader
  # train_sample, _ = iter(experiment.train_loader).next()
  data = experiment.Xtest
  data = data[torch.randperm(data.size(0))]
  inference_group_mask = (
    # sample_random_mask(batch_size, num_groups)
    torch.ones(data.size(0), num_groups)

    # only the core
    # torch.FloatTensor([[1, 0, 0, 0, 0]]).expand(data.size(0), -1)
  )
  info = experiment.vae.reconstruct(
    [Variable(x) for x in experiment.split_into_groups(data)],
    Variable(inference_group_mask)
  )
  reconstr = info['reconstructed']

  # Take the mean of p(x | z)
  reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
  reconstr_angles = experiment.split_into_joints(experiment.preprocess_inverse(reconstr_tensor))

  true_angles = experiment.split_into_joints(experiment.preprocess_inverse(data))

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
      experiment.skeleton,
      mocap_data.frame_to_xyz(experiment.skeleton, frame(i, true_angles)),
      axes=ax1
    )
    mocap_data.plot_skeleton(
      experiment.skeleton,
      mocap_data.frame_to_xyz(experiment.skeleton, frame(i, reconstr_angles)),
      axes=ax2
    )

  plt.tight_layout()
  plt.suptitle(f'Epoch {experiment.epoch}')

  torch.set_rng_state(pytorch_rng_state)
  return fig

def knn(experiment, angles, inference_group_mask, k):
  train_data = experiment.Xtrain

  # Start with a vector of all ones, use the `split_into_groups` method to get
  # things split up appropriately and then mask on `inference_group_mask`.
  mask = torch.cat([m * ones for m, ones in zip(inference_group_mask.view(-1), experiment.split_into_groups(torch.ones(1, train_data.size(1))))], dim=1)
  dists = torch.sqrt(torch.sum((train_data - angles).pow(2) * mask, dim=1))
  dists, indices = torch.sort(dists)
  return torch.mean(train_data[indices[:k]], dim=0)

# def viz_reconstruction_with_knn(experiment, plot_seed, num_examples, k):
#   pytorch_rng_state = torch.get_rng_state()
#   torch.manual_seed(plot_seed)

#   # grab random sample from train_loader
#   train_data = experiment.Xtrain
#   test_data = experiment.Xtest

#   # permute the test data so that our plots show more variety
#   test_data = test_data[torch.randperm(test_data.size(0))]

#   inference_group_mask = (
#     # sample_random_mask(batch_size, num_groups)
#     # torch.ones(test_data.size(0), num_groups)

#     # only the core
#     torch.FloatTensor([[1, 0, 0, 0, 0]])
#   )
#   info = experiment.vae.reconstruct(
#     [Variable(x) for x in experiment.split_into_groups(test_data)],
#     Variable(inference_group_mask.expand(test_data.size(0), -1))
#   )
#   reconstr = info['reconstructed']

#   # Take the mean of p(x | z)
#   reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
#   reconstr_angles = experiment.split_into_joints(experiment.preprocess_inverse(reconstr_tensor))

#   true_angles = experiment.split_into_joints(experiment.preprocess_inverse(test_data))

#   def frame(ix, angles):
#     stuff = {j: list(x[ix].numpy()) for j, x in zip(joint_order, angles)}
#     # We can't forget the (unlearned) translation dofs
#     stuff['root'] = [0, 0, 0] + stuff['root']
#     return stuff

#   fig = plt.figure(figsize=(12, 4))
#   for i in range(num_examples):
#     # I'm not aware of any GridSpec for projection='3d'
#     ax1 = fig.add_subplot(3, num_examples, i + 1, projection='3d')
#     ax2 = fig.add_subplot(3, num_examples, i + num_examples + 1, projection='3d')
#     ax3 = fig.add_subplot(3, num_examples, i + num_examples * 2 + 1, projection='3d')
#     mocap_data.plot_skeleton(
#       experiment.skeleton,
#       mocap_data.frame_to_xyz(experiment.skeleton, frame(i, true_angles)),
#       axes=ax1
#     )
#     mocap_data.plot_skeleton(
#       experiment.skeleton,
#       mocap_data.frame_to_xyz(experiment.skeleton, frame(i, reconstr_angles)),
#       axes=ax2
#     )
#     knn_angles = experiment.split_into_joints(experiment.preprocess_inverse(
#       knn(experiment, test_data[i], inference_group_mask, k).view(1, -1)
#     ))
#     mocap_data.plot_skeleton(
#       experiment.skeleton,
#       mocap_data.frame_to_xyz(experiment.skeleton, frame(0, knn_angles)),
#       axes=ax3
#     )

#   plt.tight_layout()
#   plt.suptitle(f'Epoch {experiment.epoch}')

#   torch.set_rng_state(pytorch_rng_state)
#   return fig

# # See https://stackoverflow.com/questions/1501651/log-output-of-multiprocessing-process?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# TODO
def run_experiment(dim_z, num_train_trials):
  torch.manual_seed(0)
  torch.set_num_threads(2)

  print('Running experiment...')
  print(f'  dim_z = {dim_z}')
  print(f'  num_train_trials = {num_train_trials}')

  # Subject 7 trial 12 is a faster walk so it's a bit different, best to ignore.
  experiment = MocapGroupedJMVAE(
    subject=7,
    train_trials=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:num_train_trials],
    test_trials=[11],
    dim_z=dim_z,
    batch_size=64,
    base_results_dir=Path('results/'),
    prefix='held_out_table_jmvae_'
  )

  stdout_path = experiment.results_dir / 'stdout'
  stderr_path = experiment.results_dir / 'stderr'
  with open(stdout_path, 'w') as stdout, open(stderr_path, 'w') as stderr:
    sys.stdout = stdout
    sys.stderr = stderr
    experiment.train(1000)
    print(f'Final test euclidean error: {experiment.test_euclidean_error()}')
    print(f'Final test angle error: {experiment.test_angle_error()}')
    print(f'Final test log likelihood: {experiment.test_loglik()}')

  # Reset sys.stdout and sys.stderr outside of the with in case anything goes
  # wrong
  sys.stdout = sys.__stdout__
  sys.stderr = sys.__stderr__

if __name__ == '__main__':
  dim_zs = [16]
  num_train_trials = [1, 2, 5, 10]

  # Pool will by default use as many processes as `os.cpu_count()` indicates.
  with multiprocessing.Pool(processes=7) as pool:
    pool.starmap(
      run_experiment,
      itertools.product(dim_zs, num_train_trials)
    )

# if __name__ == '__main__':
#   torch.manual_seed(0)

#   experiment = MocapGroupedJMVAE(
#     subject=7,
#     train_trials=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     test_trials=[11],
#     dim_z=16,
#     batch_size=64,
#     base_results_dir=Path('results/'),
#     prefix='deleteme_jmvae_mocap_'
#   )
#   experiment.train(1000)
#   print(f'Final test euclidean error: {experiment.test_euclidean_error()}')
#   print(f'Final test angle error: {experiment.test_angle_error()}')
#   print(f'Final test log likelihood: {experiment.test_loglik()}')
