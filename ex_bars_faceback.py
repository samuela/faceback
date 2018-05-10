import itertools
from pathlib import Path

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
from faceback import SparseProductOfExpertsVAE, OneSidedFacebackoiVAE
from utils import no_ticks


class BarsFaceback(object):
  """Runs the sparse PoE faceback framework on the bars data with exactly one
  bar present per image."""
  def __init__(
      self,
      img_size,
      num_samples,
      batch_size,
      dim_z,
      lam,
      sparsity_matrix_lr,
      initial_baseline_precision,
      base_results_dir=None
  ):
    self.img_size = img_size
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.dim_z = dim_z
    self.lam = lam
    self.sparsity_matrix_lr = sparsity_matrix_lr
    self.initial_baseline_precision = initial_baseline_precision
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

    self.epoch_counter = itertools.count()
    self.epoch = None
    self.elbo_per_iter = []
    self.test_loglik_per_iter = []

    self.prior_z = Normal(
      Variable(torch.zeros(1, dim_z)),
      Variable(torch.ones(1, dim_z))
    )
    self.baseline_precision = Variable(
      self.initial_baseline_precision * torch.ones(1),
      requires_grad=True
    )

    dim_xs = [self.img_size] * self.img_size
    # self.vae = SparseProductOfExpertsVAE(
    self.vae = OneSidedFacebackoiVAE(
      inference_nets=[self.make_inference_net(dim_x) for dim_x in dim_xs],
      generative_nets=[self.make_generative_net(dim_x) for dim_x in dim_xs],
      prior_z=self.prior_z,
      prior_theta=NormalPriorTheta(sigma=1e3),
      # prior_theta=NoPriorTheta(),
      lam=lam
    )

    self.optimizer = MetaOptimizer([
      torch.optim.Adam(
        set(p for net in self.vae.inference_nets for p in net.parameters()),
        lr=1e-3
      ),
      torch.optim.Adam(
        set(p for net in self.vae.generative_nets for p in net.parameters()),
        lr=1e-3
      ),
      torch.optim.Adam([self.baseline_precision], lr=1e-3),
      torch.optim.SGD([self.vae.sparsity_matrix], lr=self.sparsity_matrix_lr)
    ])

    # self.elbo_plot_fig, self.elbo_plot_ax = None, None
    # self.reconstruction_fig, self.reconstruction_ax = None, None
    # self.sparsity_fig, self.sparsity_ax = None, None
    # self.sparsity_colorbar = None

    if self.base_results_dir is not None:
      self._init_results_dir()

  def sample_data(self, num_samples, noise_stddev=0.01):
    """Sample a bars image. Produces a Tensor of shape [num_samples,
    self.img_size, self.img_size]."""
    # return (
    #   sample_many_one_bar_images(num_samples, self.img_size) +
    #   noise_stddev * torch.randn(num_samples, self.img_size, self.img_size)
    # )
    return (
      sample_many_bars_images(num_samples, self.img_size, 0.5 * torch.ones(self.img_size), torch.zeros(self.img_size)) +
      noise_stddev * torch.randn(num_samples, self.img_size, self.img_size)
    )

  def make_generative_net(self, dim_x):
    return NormalNet(
      torch.nn.Linear(self.dim_z, dim_x),
      torch.nn.Sequential(
        torch.nn.Linear(self.dim_z, dim_x),
        Lambda(lambda x: torch.exp(0.5 * x))
      )
      # Lambda(lambda x: Variable(torch.ones(x.size(0), dim_x)))
    )

  def make_inference_net(self, dim_x):
    return Normal_MeanPrecisionNet(
      torch.nn.Linear(dim_x, self.dim_z),
      torch.nn.Sequential(
        torch.nn.Linear(dim_x, self.dim_z),
        Lambda(lambda x: torch.exp(x) + self.baseline_precision)
      )
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
            # sample_random_mask(actual_batch_size, self.img_size)
            torch.ones(actual_batch_size, self.img_size)
            # torch.FloatTensor([1, 0, 0, 0]).expand(actual_batch_size, -1)
          )
        )
        elbo = info['elbo']
        z_kl = info['z_kl']
        reconstruction_log_likelihood = info['reconstruction_log_likelihood']
        logprob_theta = info['logprob_theta']
        logprob_L1 = info['logprob_L1']
        test_ll = self.test_loglik().data[0]

        self.optimizer.zero_grad()
        (-elbo).backward()
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
        print(self.baseline_precision.data[0])
        # print(self.vae.generative_nets[0].param_nets[1][0].bias.data)

      if self.base_results_dir is not None or True:
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

        # plt.show()

  def test_loglik(self):
    Xs = [Variable(self.test_data[:, i]) for i in range(self.img_size)]
    group_mask = Variable(torch.ones(self.test_data.size(0), self.img_size))
    q_z = self.vae.approx_posterior(Xs, group_mask)
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
    plt.imshow(self.vae.sparsity_matrix.data.numpy())
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

  def results_dir(self):
    params = ['img_size', 'num_samples', 'batch_size', 'dim_z', 'lam', 'sparsity_matrix_lr', 'initial_baseline_precision']
    poop = ' '.join([f'{p}={getattr(self, p)}' for p in params])
    return self.base_results_dir / f'faceback-oivae bars {poop}'

  def _init_results_dir(self):
    self.results_dir_elbo = self.results_dir() / 'elbo_plot'
    self.results_dir_sparsity_matrix = self.results_dir() / 'sparsity_matrix'
    self.results_dir_reconstructions = self.results_dir() / 'reconstructions'

    # The results_dir should be unique
    self.results_dir().mkdir(exist_ok=False)
    self.results_dir_elbo.mkdir(exist_ok=False)
    self.results_dir_sparsity_matrix.mkdir(exist_ok=False)
    self.results_dir_reconstructions.mkdir(exist_ok=False)

if __name__ == '__main__':
  torch.manual_seed(0)

  experiment = BarsFaceback(
    img_size=2,
    num_samples=10000,
    batch_size=32,
    dim_z=2,
    lam=0,
    sparsity_matrix_lr=1e-3,
    initial_baseline_precision=100,
    base_results_dir=Path('results/')
  )
  experiment.train(100)
