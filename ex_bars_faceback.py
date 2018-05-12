"""Running Faceback on the standard bars example."""

import itertools
import json
from pathlib import Path
import random
import string

import dill

import matplotlib.pyplot as plt

# This is actually required to use `torch.utils.data`.
# pylint: disable=unused-import
import torchvision
import torch
from torch.autograd import Variable

from kindling.distributions import (
  BernoulliNet,
  Normal,
  NormalNet,
  Normal_MeanPrecisionNet
)
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer, NoPriorTheta

from bars_data import sample_many_one_bar_images, sample_many_bars_images
from faceback import SparseProductOfExpertsVAE, OneSidedFacebackoiVAE, FacebackVAE, FacebackInferenceNet, PoopGenerativeNet
from utils import no_ticks, sample_random_mask


class BarsFaceback(object):
  """Runs the sparse PoE faceback framework on the bars data with the split group sparsity prior."""

  PARAMS = [
    'img_size',
    'num_samples',
    'batch_size',
    'dim_z',
    'lam',
    'sparsity_matrix_lr',
    'initial_baseline_precision',
    'inference_net_output_dim',
    'generative_net_input_dim',
    'noise_stddev',
    'group_available_prob',
    'initial_sigma_adjustment'
  ]

  def __init__(
      self,
      img_size,
      num_samples,
      batch_size,
      dim_z,
      lam,
      sparsity_matrix_lr,
      initial_baseline_precision,
      inference_net_output_dim,
      generative_net_input_dim,
      noise_stddev,
      group_available_prob,
      initial_sigma_adjustment,
      base_results_dir=None
  ):
    self.img_size = img_size
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.dim_z = dim_z
    self.lam = lam
    self.sparsity_matrix_lr = sparsity_matrix_lr
    self.initial_baseline_precision = initial_baseline_precision
    self.inference_net_output_dim = inference_net_output_dim
    self.generative_net_input_dim = generative_net_input_dim
    self.noise_stddev = noise_stddev
    self.group_available_prob = group_available_prob
    self.initial_sigma_adjustment = initial_sigma_adjustment
    self.base_results_dir = base_results_dir

    # Sample the training data and set up a DataLoader
    self.train_data = self.sample_data(self.num_samples)
    self.test_data = self.sample_data(1000)
    self.train_loader = torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(
        self.train_data,
        torch.zeros(self.num_samples)
      ),
      batch_size=batch_size,
      shuffle=True
    )

    self.generative_sigma_adjustment = Variable(
      self.initial_sigma_adjustment * torch.ones(1),
      requires_grad=True
    )

    self.epoch_counter = itertools.count()
    self.epoch = None
    self.elbo_per_iter = []
    self.test_loglik_per_iter = []

    self.prior_z = Normal(
      Variable(torch.zeros(1, dim_z)),
      Variable(torch.ones(1, dim_z))
    )

    dim_xs = [self.img_size] * self.img_size
    self.inference_net = FacebackInferenceNet(
      almost_inference_nets=[self.make_almost_inference_net(dim_x) for dim_x in dim_xs],
      net_output_dim=self.inference_net_output_dim,
      prior_z=self.prior_z,
      initial_baseline_precision=self.initial_baseline_precision
    )
    self.generative_net = PoopGenerativeNet(
      almost_generative_nets=[self.make_almost_generative_net(dim_x) for dim_x in dim_xs],
      net_input_dim=self.generative_net_input_dim,
      dim_z=self.dim_z
    )
    self.vae = FacebackVAE(
      inference_net=self.inference_net,
      generative_net=self.generative_net,
      prior_z=self.prior_z,
      prior_theta=NormalPriorTheta(sigma=1e3),
      lam=self.lam
    )

    # group lasso version
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
      torch.optim.SGD([self.generative_net.connectivity_matrices], lr=self.sparsity_matrix_lr),
      torch.optim.Adam([self.generative_sigma_adjustment], lr=1e-3)
    ])

    if self.base_results_dir is not None:
      # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
      self.results_folder_name = 'bars' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
      self.results_dir = self.base_results_dir / self.results_folder_name
      self._init_results_dir()

  def sample_data(self, num_samples):
    """Sample a bars image. Produces a Tensor of shape [num_samples,
    self.img_size, self.img_size]."""
    # return (
    #   sample_many_one_bar_images(num_samples, self.img_size) +
    #   noise_stddev * torch.randn(num_samples, self.img_size, self.img_size)
    # )
    return (
      sample_many_bars_images(num_samples, self.img_size, 0.5 * torch.ones(self.img_size), torch.zeros(self.img_size)) +
      self.noise_stddev * torch.randn(num_samples, self.img_size, self.img_size)
    )

  def make_almost_generative_net(self, dim_x):
    return NormalNet(
      torch.nn.Linear(self.generative_net_input_dim, dim_x),
      torch.nn.Sequential(
        torch.nn.Linear(self.generative_net_input_dim, dim_x),
        Lambda(lambda x: torch.exp(0.5 * x + self.generative_sigma_adjustment))
      )
    )

  def make_almost_inference_net(self, dim_x):
    return torch.nn.Sequential(
      torch.nn.Linear(dim_x, self.inference_net_output_dim),
      torch.nn.ReLU()
    )

  def train(self, num_epochs):
    for self.epoch in itertools.islice(self.epoch_counter, num_epochs):
      for batch_idx, (data, _) in enumerate(self.train_loader):
        # The final batch may not have the same size as `batch_size`.
        actual_batch_size = data.size(0)

        info = self.vae.elbo(
          Xs=[Variable(data[:, i]) for i in range(self.img_size)],
          group_mask=Variable(torch.ones(actual_batch_size, self.img_size)),
          inference_group_mask=Variable(
            sample_random_mask(actual_batch_size, self.img_size, self.group_available_prob)
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
        print(f'  test log lik.      : {test_ll}')
        print(self.inference_net.baseline_precision.data[0])
        print(self.generative_sigma_adjustment.data[0])
        # print(self.vae.generative_nets[0].param_nets[1][0].bias.data)

      if self.base_results_dir is not None:
        # This has a good mix when img_size = 4
        fig = self.viz_reconstruction(12)
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
        self.viz_reconstruction(12)
        self.viz_elbo()
        self.viz_sparsity()
        plt.show()

  def test_loglik(self):
    Xs = [Variable(self.test_data[:, i]) for i in range(self.img_size)]
    group_mask = Variable(torch.ones(self.test_data.size(0), self.img_size))
    q_z = self.inference_net(Xs, group_mask)
    return self.vae.log_likelihood(Xs, group_mask, q_z.sample())

  def viz_elbo(self):
    fig = plt.figure()
    plt.plot(self.elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    return fig

  def viz_sparsity(self):
    """Visualize the sparisty matrix associating latent components with
    groups."""
    fig = plt.figure()
    group_norms = torch.sqrt(
      torch.sum(self.inference_net.precision_layers.data.pow(2), dim=1) +
      torch.sum(self.generative_net.connectivity_matrices.data.pow(2), dim=2)
    )
    plt.imshow(group_norms.numpy())
    plt.colorbar()
    plt.xlabel('latent components')
    plt.ylabel('groups')
    return fig

  def viz_reconstruction(self, plot_seed):
    pytorch_rng_state = torch.get_rng_state()
    torch.manual_seed(plot_seed)

    # grab random sample from train_loader
    train_sample, _ = iter(self.train_loader).next()
    inference_group_mask = Variable(
      # sample_random_mask(self.batch_size, self.img_size)
      torch.ones(self.batch_size, self.img_size)
      # torch.FloatTensor([1, 0, 0, 0]).expand(self.batch_size, -1)
    )
    info = self.vae.reconstruct(
      [Variable(train_sample[:, i]) for i in range(self.img_size)],
      inference_group_mask
    )
    reconstr = info['reconstructed']
    reconstr_tensor = torch.stack([reconstr[i].mu.data for i in range(self.img_size)], dim=1)

    fig, ax = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
      ax[0, i].imshow(train_sample[i].numpy(), vmin=0, vmax=1)
      ax[1, i].imshow(reconstr_tensor[i].numpy(), vmin=0, vmax=1)

      no_ticks(ax[0, i])
      no_ticks(ax[1, i])

    ax[0, 0].set_ylabel('true')
    ax[1, 0].set_ylabel('reconst')

    plt.tight_layout()
    plt.suptitle(f'Epoch {self.epoch}')

    torch.set_rng_state(pytorch_rng_state)
    return fig

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
      {p: getattr(self, p) for p in BarsFaceback.PARAMS},
      open(self.results_dir_params, 'w'),
      sort_keys=True,
      indent=2,
      separators=(',', ': ')
    )

if __name__ == '__main__':
  torch.manual_seed(0)

  experiment = BarsFaceback(
    img_size=2,
    num_samples=10000,
    batch_size=32,
    dim_z=4,
    lam=0,
    sparsity_matrix_lr=1e-3,
    initial_baseline_precision=100,
    inference_net_output_dim=8,
    generative_net_input_dim=8,
    noise_stddev=0.05,
    group_available_prob=0.5,
    initial_sigma_adjustment=1,
    base_results_dir=Path('results/')
  )
  experiment.train(250)
