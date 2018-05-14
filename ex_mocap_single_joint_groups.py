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

from faceback import FacebackGenerativeNet, FacebackInferenceNet, FacebackVAE
from kindling.distributions import Normal, NormalNet
from kindling.utils import Lambda, MetaOptimizer, NormalPriorTheta
from mocap import mocap_data
from utils import sample_random_mask

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

class MocapSingleJointGroupsFaceback(object):
  """Experiment with mocap data running each joint as its own faceback group."""

  PARAMS = [
    'subject',
    'train_trials',
    'test_trials',
    'dim_z',
    'batch_size',
    'lam',
    'sparsity_matrix_lr',
    'inference_net_output_dim',
    'generative_net_input_dim',
    'initial_baseline_precision',
    'prior_theta_sigma',
    'group_available_prob'
  ]

  def __init__(
      self,
      subject,
      train_trials,
      test_trials,
      dim_z,
      batch_size,
      lam,
      sparsity_matrix_lr,
      inference_net_output_dim,
      generative_net_input_dim,
      initial_baseline_precision,
      prior_theta_sigma,
      group_available_prob,
      base_results_dir=None
  ):
    self.subject = subject
    self.train_trials = train_trials
    self.test_trials = test_trials
    self.dim_z = dim_z
    self.batch_size = batch_size
    self.lam = lam
    self.sparsity_matrix_lr = sparsity_matrix_lr
    self.inference_net_output_dim = inference_net_output_dim
    self.generative_net_input_dim = generative_net_input_dim
    self.initial_baseline_precision = initial_baseline_precision
    self.prior_theta_sigma = prior_theta_sigma
    self.group_available_prob = group_available_prob
    self.base_results_dir = base_results_dir

    self.epoch_counter = itertools.count()
    self.epoch = None
    self.elbo_per_iter = []
    self.test_loglik_per_iter = []

    self.load_data()

    self.prior_z = Normal(
      Variable(torch.zeros(1, dim_z)),
      Variable(torch.ones(1, dim_z))
    )

    self.inference_net = FacebackInferenceNet(
      almost_inference_nets=[self.make_almost_inference_net(self.joint_dims[j]) for j in joint_order],
      net_output_dim=self.inference_net_output_dim,
      prior_z=self.prior_z,
      initial_baseline_precision=self.initial_baseline_precision
    )
    self.generative_net = FacebackGenerativeNet(
      almost_generative_nets=[self.make_almost_generative_net(self.joint_dims[j]) for j in joint_order],
      net_input_dim=self.generative_net_input_dim,
      dim_z=self.dim_z
    )
    self.vae = FacebackVAE(
      inference_net=self.inference_net,
      generative_net=self.generative_net,
      prior_z=self.prior_z,
      prior_theta=NormalPriorTheta(sigma=self.prior_theta_sigma),
      lam=self.lam
    )

    self.optimizer = MetaOptimizer([
      # Inference parameters
      torch.optim.Adam(
        set(p for net in self.inference_net.almost_inference_nets for p in net.parameters()),
        lr=1e-3
      ),
      torch.optim.Adam([self.inference_net.mu_layers], lr=1e-3),
      torch.optim.SGD([self.inference_net.precision_layers], lr=self.sparsity_matrix_lr),
      torch.optim.Adam([self.inference_net.baseline_precision], lr=1e-3),

      # Generative parameters
      torch.optim.Adam(
        set(p for net in self.generative_net.almost_generative_nets for p in net.parameters()),
        lr=1e-3
      ),
      torch.optim.SGD([self.generative_net.connectivity_matrices], lr=self.sparsity_matrix_lr)
    ])

    if self.base_results_dir is not None:
      # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
      self.results_folder_name = 'shallow_model_mocap_single_joint' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
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
          inference_group_mask=Variable(
            sample_random_mask(actual_batch_size, num_groups, self.group_available_prob)
          )
        )
        elbo = info['elbo']
        loss = info['loss']
        z_kl = info['z_kl']
        reconstruction_log_likelihood = info['reconstruction_log_likelihood']
        logprob_theta = info['logprob_theta']
        logprob_L1 = info['logprob_L1']
        test_ll = self.test_loglik().data[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.vae.proximal_step(self.sparsity_matrix_lr * self.lam)

        self.elbo_per_iter.append(elbo.data[0])
        self.test_loglik_per_iter.append(test_ll)
        print(f'Epoch {self.epoch}, {batch_idx} / {len(self.train_loader)}')
        print(f'  ELBO: {elbo.data[0]}')
        print(f'    -KL(q(z) || p(z)): {-z_kl.data[0]}')
        print(f'    loglik_term      : {reconstruction_log_likelihood.data[0]}')
        print(f'    log p(theta)     : {logprob_theta.data[0]}')
        print(f'    L1               : {logprob_L1.data[0]}')
        print(f'  test log lik.      : {test_ll}', flush=True)

      # Checkpoint every 10 epochs
      if self.epoch % 50 == 0:
        self.checkpoint()

    # Checkpoint at the very end as well
    self.checkpoint()

  def make_almost_generative_net(self, dim_x):
    return NormalNet(
      torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(self.generative_net_input_dim, dim_x)
      ),
      torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(self.generative_net_input_dim, dim_x),
        # Learn the log variance
        Lambda(lambda x: torch.exp(0.5 * x))
      )
    )

  def make_almost_inference_net(self, dim_x):
    hidden_size = 32
    return torch.nn.Sequential(
      torch.nn.Linear(dim_x, hidden_size),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size, self.inference_net_output_dim),
      torch.nn.ReLU()
    )

  def viz_elbo(self):
    """ELBO per iteration"""
    fig = plt.figure()
    plt.plot(self.elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    return fig

  def viz_sparsity(self):
    """Visualize the sparisty matrix associating latent components with
    groups."""
    fig = plt.figure()
    plt.imshow(self.vae.sparsity_matrix().data.numpy())
    plt.colorbar()
    plt.xlabel('latent components')
    plt.ylabel('groups')
    return fig

  def viz_reconstruction(self, plot_seed, num_examples):
    pytorch_rng_state = torch.get_rng_state()
    torch.manual_seed(plot_seed)

    # grab random sample from train_loader
    train_sample, _ = iter(self.train_loader).next()
    inference_group_mask = Variable(
      # sample_random_mask(batch_size, num_groups)
      torch.ones(self.batch_size, num_groups)
    )
    info = self.vae.reconstruct(
      [Variable(x) for x in self.split_into_groups(train_sample)],
      inference_group_mask
    )
    reconstr = info['reconstructed']

    true_angles = self.split_into_groups(self.preprocess_inverse(train_sample))

    # Take the mean of p(x | z)
    reconstr_tensor = torch.cat([reconstr[i].mu.data for i in range(num_groups)], dim=1)
    reconstr_angles = self.split_into_groups(self.preprocess_inverse(reconstr_tensor))

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
        self.skeleton,
        mocap_data.frame_to_xyz(self.skeleton, frame(i, true_angles)),
        axes=ax1
      )
      mocap_data.plot_skeleton(
        self.skeleton,
        mocap_data.frame_to_xyz(self.skeleton, frame(i, reconstr_angles)),
        axes=ax2
      )

    plt.tight_layout()
    plt.suptitle(f'Epoch {self.epoch}')

    torch.set_rng_state(pytorch_rng_state)
    return fig

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
    Xs = [Variable(x) for x in self.split_into_groups(self.Xtest)]
    group_mask = Variable(torch.ones(self.Xtest.size(0), num_groups))
    q_z = self.inference_net(Xs, group_mask)
    return self.vae.log_likelihood(Xs, group_mask, q_z.sample())

  def preprocess(self, x):
    """Preprocess the angular data to lie between 0 and 1."""
    # Some of these things aren't used, and we don't want to divide by zero
    return (x - self.angular_mins) / torch.clamp(self.angular_maxs - self.angular_mins, min=0.1)

  def preprocess_inverse(self, y):
    """Inverse of `preprocess`."""
    return y * torch.clamp(self.angular_maxs - self.angular_mins, min=0.1) + self.angular_mins

  def split_into_groups(self, data):
    poop = np.cumsum([0] + [self.joint_dims[j] for j in joint_order])
    return [data[:, poop[i]:poop[i + 1]] for i in range(num_groups)]

  def _init_results_dir(self):
    self.results_dir_params = self.results_dir / 'params.json'
    self.results_dir_elbo = self.results_dir / 'elbo_plot'
    self.results_dir_sparsity_matrix = self.results_dir / 'sparsity_matrix'
    self.results_dir_reconstructions = self.results_dir / 'reconstructions'
    self.results_dir_pickles = self.results_dir / 'pickles'

    # The results_dir should be unique
    self.results_dir.mkdir(exist_ok=False)
    self.results_dir_elbo.mkdir(exist_ok=False)
    self.results_dir_sparsity_matrix.mkdir(exist_ok=False)
    self.results_dir_reconstructions.mkdir(exist_ok=False)
    self.results_dir_pickles.mkdir(exist_ok=False)
    json.dump(
      {p: getattr(self, p) for p in MocapSingleJointGroupsFaceback.PARAMS},
      open(self.results_dir_params, 'w'),
      sort_keys=True,
      indent=2,
      separators=(',', ': ')
    )

  def checkpoint(self):
    if self.base_results_dir is not None:
      fig = self.viz_reconstruction(0, num_examples=6)
      plt.savefig(self.results_dir_reconstructions / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig = self.viz_elbo()
      plt.savefig(self.results_dir_elbo / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig = self.viz_sparsity()
      plt.savefig(self.results_dir_sparsity_matrix / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      dill.dump(self, open(self.results_dir_pickles / f'epoch{self.epoch}.p', 'wb'))
    else:
      self.viz_reconstruction(0, num_examples=6)
      self.viz_elbo()
      self.viz_sparsity()
      plt.show()

# See https://stackoverflow.com/questions/1501651/log-output-of-multiprocessing-process?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def run_experiment(lam, group_available_prob, dim_z):
  print('Running experiment...')
  print(f'  lam = {lam}')
  print(f'  group_available_prob = {group_available_prob}')
  print(f'  dim_z = {dim_z}')

  # Subject 7 trial 12 is a faster walk so it's a bit different, best to ignore.
  experiment = MocapSingleJointGroupsFaceback(
    subject=55,
    train_trials=list(range(1, 25 + 1)),
    test_trials=[26, 27, 28],
    dim_z=dim_z,
    batch_size=64,
    lam=lam,
    sparsity_matrix_lr=1e-3,
    inference_net_output_dim=32,
    generative_net_input_dim=32,
    initial_baseline_precision=100,
    prior_theta_sigma=1,
    group_available_prob=group_available_prob,
    base_results_dir=Path('results/')
  )

  stdout_path = experiment.results_dir / 'stdout'
  stderr_path = experiment.results_dir / 'stderr'
  with open(stdout_path, 'w') as stdout, open(stderr_path, 'w') as stderr:
    sys.stdout = stdout
    sys.stderr = stderr
    experiment.train(1000)

  # Reset sys.stdout and sys.stderr outside of the with in case anything goes
  # wrong
  sys.stdout = sys.__stdout__
  sys.stderr = sys.__stderr__

if __name__ == '__main__':
  lams = [0.1, 1]
  group_available_probs = [0.5, 1]
  dim_zs = [4, 8, 16]

  # Pool will by default use as many processes as `os.cpu_count()` indicates.
  with multiprocessing.Pool(processes=2) as pool:
    pool.starmap(
      run_experiment,
      itertools.product(lams, group_available_probs, dim_zs)
    )
