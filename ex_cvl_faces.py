import itertools
import json
import random
import string
import time
from pathlib import Path

import dill

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torchvision

from skimage import io

from kindling.distributions import Normal, NormalNet
from kindling.utils import Lambda, NormalPriorTheta, MetaOptimizer

from faceback import FacebackVAE, FacebackInferenceNet, FacebackGenerativeNet
from utils import no_ticks, sample_random_mask, viz_sparsity

# From readme.txt:
# 17+ => more pictures (+2)
# 35- => one picture is missing
# 44- => one picture is missing
# 93- => one picture is missing
class CVLDataset(torch.utils.data.Dataset):
  def __init__(self, path, subjects=range(1, 114 + 1), transform=None):
    self.path = path
    self.subjects = subjects
    self.transform = transform

    self.cache = [None] * len(self.subjects)

  def __len__(self):
    return len(self.subjects)

  def __getitem__(self, index):
    if self.cache[index] is not None:
      return self.cache[index]
    else:
      subject = self.subjects[index]

      face_right = self.path / str(subject) / 'MVC-001F.JPG'
      face_halfright = self.path / str(subject) / 'MVC-002F.JPG'
      face_front = self.path / str(subject) / 'MVC-003F.JPG'
      face_halfleft = self.path / str(subject) / 'MVC-004F.JPG'
      face_left = self.path / str(subject) / 'MVC-005F.JPG'
      face_smile = self.path / str(subject) / 'MVC-006F.JPG'
      face_teethsmile = self.path / str(subject) / 'MVC-007F.JPG'
      paths = [
        face_right,
        face_halfright,
        face_front,
        face_halfleft,
        face_left,
        face_smile,
        face_teethsmile,
      ]

      # This is kind of a hack since we're assuming that the processed images finish as 64x64
      blank = torch.zeros(1, 64, 64)

      mask = [path.exists() for path in paths]
      images = [(io.imread(p) if mask[i] else blank) for i, p in enumerate(paths)]
      if self.transform is not None:
        images = [(self.transform(im) if mask[i] else im) for i, im in enumerate(images)]

      res = [torch.FloatTensor(mask)] + images
      self.cache[index] = res
      return res

# There are seven groups in this dataset:
#   * right
#   * halfright
#   * front
#   * halfleft
#   * left
#   * smile
#   * teethsmile
num_groups = 7

transform = torchvision.transforms.Compose([
  torchvision.transforms.ToPILImage(),
  torchvision.transforms.CenterCrop(450),
  torchvision.transforms.Grayscale(),
  torchvision.transforms.Resize(64),
  torchvision.transforms.ToTensor(),
  # Normalize assuming the mean is 0.5 and std dev is 0.5.
  torchvision.transforms.Normalize((0.5,), (0.5,))
])

class CVLFacebackExperiment(object):

  PARAMS = [
    # 'img_size',
    'dim_z',
    'batch_size',
    'lam',
    'sparsity_matrix_lr',
    'inference_net_output_dim',
    'generative_net_input_dim',
    'initial_baseline_precision',
    'prior_theta_sigma',
    'group_available_prob',
    'inference_net_num_filters',
    'generative_net_num_filters',
    'use_gpu',
  ]

  def __init__(
      self,
      # img_size,
      dim_z,
      batch_size,
      lam,
      sparsity_matrix_lr,
      inference_net_output_dim,
      generative_net_input_dim,
      initial_baseline_precision,
      prior_theta_sigma,
      group_available_prob,
      inference_net_num_filters,
      generative_net_num_filters,
      use_gpu,
      base_results_dir=None,
      prefix='faces_'
  ):
    # self.img_size = img_size
    self.dim_z = dim_z
    self.batch_size = batch_size
    self.lam = lam
    self.sparsity_matrix_lr = sparsity_matrix_lr
    self.inference_net_output_dim = inference_net_output_dim
    self.generative_net_input_dim = generative_net_input_dim
    self.initial_baseline_precision = initial_baseline_precision
    self.prior_theta_sigma = prior_theta_sigma
    self.group_available_prob = group_available_prob
    self.inference_net_num_filters = inference_net_num_filters
    self.generative_net_num_filters = generative_net_num_filters
    self.use_gpu = use_gpu
    self.base_results_dir = base_results_dir
    self.prefix = prefix

    self.epoch_counter = itertools.count()
    self.epoch = None
    self.elbo_per_iter = []

    self.load_data()

    if self.use_gpu:
      self.prior_z = Normal(
        Variable(torch.zeros(1, dim_z).cuda()),
        Variable(torch.ones(1, dim_z).cuda())
      )
    else:
      self.prior_z = Normal(
        Variable(torch.zeros(1, dim_z)),
        Variable(torch.ones(1, dim_z))
      )

    self.inference_net = FacebackInferenceNet(
      almost_inference_nets=[
        self.make_almost_inference_net(64 * 64)
        for _ in range(num_groups)
      ],
      net_output_dim=self.inference_net_output_dim,
      prior_z=self.prior_z,
      initial_baseline_precision=self.initial_baseline_precision,
      use_gpu=self.use_gpu
    )
    self.generative_net = FacebackGenerativeNet(
      almost_generative_nets=[
        self.make_almost_generative_net(64 * 64)
        for _ in range(num_groups)
      ],
      net_input_dim=self.generative_net_input_dim,
      dim_z=self.dim_z,
      use_gpu=self.use_gpu
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
      torch.optim.Adam(
        [net.sigma_net.extra_args[0] for net in self.generative_net.almost_generative_nets],
        lr=1e-3
      ),
      torch.optim.SGD([self.generative_net.connectivity_matrices], lr=self.sparsity_matrix_lr)
    ])

    if self.base_results_dir is not None:
      # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
      self.results_folder_name = self.prefix + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
      self.results_dir = self.base_results_dir / self.results_folder_name
      self._init_results_dir()

  def train(self, num_epochs):
    for self.epoch in itertools.islice(self.epoch_counter, num_epochs):
      for batch_idx, batch in enumerate(self.train_loader):
        group_mask = batch[0]
        views = batch[1:]

        # The final batch may not have the same size as `batch_size`.
        actual_batch_size = group_mask.size(0)

        # Multiply in the group mask because we don't want to use things for
        # inference that aren't available in the data at all.
        inference_group_mask = (
          sample_random_mask(actual_batch_size, num_groups, self.group_available_prob) *
          group_mask
        )
        if self.use_gpu:
          group_mask = group_mask.cuda()
          inference_group_mask = inference_group_mask.cuda()
          views = [x.cuda() for x in views]

        info = self.vae.elbo(
          Xs=[Variable(x) for x in views],
          group_mask=Variable(group_mask),
          inference_group_mask=Variable(inference_group_mask)
        )
        elbo = info['elbo']
        loss = info['loss']
        z_kl = info['z_kl']
        reconstruction_log_likelihood = info['reconstruction_log_likelihood']
        logprob_theta = info['logprob_theta']
        logprob_L1 = info['logprob_L1']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.vae.proximal_step(self.sparsity_matrix_lr * self.lam)

        self.elbo_per_iter.append(elbo.data[0])
        print(f'Epoch {self.epoch}, {batch_idx} / {len(self.train_loader)}')
        print(f'  ELBO: {elbo.data[0]}')
        print(f'    -KL(q(z) || p(z)): {-z_kl.data[0]}')
        print(f'    loglik_term      : {reconstruction_log_likelihood.data[0]}')
        print(f'    log p(theta)     : {logprob_theta.data[0]}')
        print(f'    L1               : {logprob_L1.data[0]}')

      # Checkpoint every once in a while
      if self.epoch % 50 == 0:
        self.checkpoint()

    # Checkpoint at the very end as well
    self.checkpoint()

  def make_almost_generative_net(self, dim_x):
    # We learn a std dev for each pixel which is not a function of the input.
    # Note that this Variable is NOT going to show up in `net.parameters()` and
    # therefore it is implicitly free from the ridge penalty/p(theta) prior.
    init_log_sigma = torch.log(1e-2 * torch.ones(1, 1, 64, 64))

    # See https://github.com/pytorch/examples/blob/master/dcgan/main.py#L107
    dim_in = self.generative_net_input_dim
    ngf = self.generative_net_num_filters
    model = torch.nn.Sequential(
      Lambda(lambda x: x.view(-1, dim_in, 1, 1)),
      torch.nn.ConvTranspose2d( dim_in, ngf * 8, 4, 1, 0, bias=False),
      torch.nn.BatchNorm2d(ngf * 8),
      torch.nn.ReLU(inplace=True),
      # state size. (ngf*8) x 4 x 4
      torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(ngf * 4),
      torch.nn.ReLU(inplace=True),
      # state size. (ngf*4) x 8 x 8
      torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(ngf * 2),
      torch.nn.ReLU(inplace=True),
      # state size. (ngf*2) x 16 x 16
      torch.nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(ngf),
      torch.nn.ReLU(inplace=True),
      # state size. (ngf) x 32 x 32
      torch.nn.ConvTranspose2d(    ngf,       1, 4, 2, 1, bias=False),
      # state size. 1 x 64 x 64
      torch.nn.Tanh()
    )

    if self.use_gpu:
      model = model.cuda()
      init_log_sigma = init_log_sigma.cuda()

    log_sigma = Variable(init_log_sigma, requires_grad=True)
    return NormalNet(
      model,
      Lambda(
        lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1, -1, -1)) + 1e-3,
        extra_args=(log_sigma,)
      )
    )

  def make_almost_inference_net(self, dim_x):
    ndf = self.inference_net_num_filters
    model = torch.nn.Sequential(
      # input is (nc) x 64 x 64
      torch.nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
      torch.nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(ndf * 2),
      torch.nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(ndf * 4),
      torch.nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8

      # Flatten the filter channels
      Lambda(lambda x: x.view(x.size(0), -1)),

      torch.nn.Linear((ndf * 4) * 8 * 8, self.inference_net_output_dim),
      torch.nn.LeakyReLU(0.2, inplace=True)

      # This is the rest of the DCGAN discriminator
      # torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      # torch.nn.BatchNorm2d(ndf * 8),
      # torch.nn.LeakyReLU(0.2, inplace=True),
      # # state size. (ndf*8) x 4 x 4
      # torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
      # torch.nn.Sigmoid()
    )

    return (model.cuda() if self.use_gpu else model)

  def load_data(self):
    path = Path('data/cvl_faces/')
    # all_subjects = range(1, 114 + 1)
    train_subjects = range(1, 100 + 1)
    test_subjects = range(101, 114 + 1)

    self.train_dataset = CVLDataset(path, subjects=train_subjects, transform=transform)
    self.train_loader = torch.utils.data.DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True
    )

    self.test_dataset = CVLDataset(path, subjects=test_subjects, transform=transform)
    self.test_loader = torch.utils.data.DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=True
    )

  def checkpoint(self):
    if self.base_results_dir is not None:
      fig = viz_reconstruction_all_views(self, self.train_dataset, range(8))
      plt.savefig(self.results_dir_train_reconstructions / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig = viz_reconstruction_all_views(self, self.test_dataset, range(8))
      plt.savefig(self.results_dir_test_reconstructions / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      # 35, 44, 93 are all missing the last smiling with teeth face. Don't
      # forget the off-by-one error though!
      fig = viz_reconstruction_all_views(self, self.train_dataset, [34, 43, 92])
      plt.savefig(self.results_dir_train_missing_reconstructions / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig = viz_elbo(self)
      plt.savefig(self.results_dir_elbo / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      fig, _ = viz_sparsity(self.vae)
      plt.savefig(self.results_dir_sparsity_matrix / f'epoch{self.epoch}.pdf')
      plt.close(fig)

      # t0 = time.time()
      # dill.dump(self, open(self.results_dir_pickles / f'epoch{self.epoch}.p', 'wb'))
      # print(f'(dilling took {time.time() - t0} seconds.)')
    else:
      # viz_reconstruction_all_views(self, range(8))
      # viz_elbo(self)
      # viz_sparsity(self.vae)
      # plt.show()
      pass

  def _init_results_dir(self):
    self.results_dir_params = self.results_dir / 'params.json'
    self.results_dir_elbo = self.results_dir / 'elbo_plot'
    self.results_dir_sparsity_matrix = self.results_dir / 'sparsity_matrix'
    self.results_dir_test_reconstructions = self.results_dir / 'test_reconstructions'
    self.results_dir_train_reconstructions = self.results_dir / 'train_reconstructions'
    self.results_dir_train_missing_reconstructions = self.results_dir / 'train_missing_reconstructions'
    self.results_dir_pickles = self.results_dir / 'pickles'

    # The results_dir should be unique
    self.results_dir.mkdir(exist_ok=False)
    self.results_dir_elbo.mkdir(exist_ok=False)
    self.results_dir_sparsity_matrix.mkdir(exist_ok=False)
    self.results_dir_test_reconstructions.mkdir(exist_ok=False)
    self.results_dir_train_reconstructions.mkdir(exist_ok=False)
    self.results_dir_train_missing_reconstructions.mkdir(exist_ok=False)
    self.results_dir_pickles.mkdir(exist_ok=False)
    json.dump(
      {p: getattr(self, p) for p in CVLFacebackExperiment.PARAMS},
      open(self.results_dir_params, 'w'),
      sort_keys=True,
      indent=2,
      separators=(',', ': ')
    )

def viz_elbo(experiment):
  """ELBO per iteration"""
  fig = plt.figure()
  plt.plot(experiment.elbo_per_iter)
  plt.xlabel('iteration')
  plt.ylabel('ELBO')
  return fig

def viz_reconstruction(experiment, indices):
  num_examples = len(indices)
  fig, ax = plt.subplots(3, num_examples, figsize=(12, 4))

  for i, subj in enumerate(indices):
    batch = experiment.test_dataset[subj]
    group_mask = batch[0]
    views = batch[1:]

    # front view
    ax[0, i].imshow(views[2].squeeze(), cmap='gray')
    # face right
    ax[1, i].imshow(views[0].squeeze(), cmap='gray')

    inference_group_mask = (
      # sample_random_mask(1, num_groups)
      # torch.ones(1, num_groups)
      torch.FloatTensor([[0, 1, 1, 1, 1, 1, 1]])
    )
    views = [x.unsqueeze(0) for x in views]
    if experiment.use_gpu:
      inference_group_mask = inference_group_mask.cuda()
      views = [x.cuda() for x in views]

    info = experiment.vae.reconstruct(
      [Variable(x) for x in views],
      Variable(inference_group_mask)
    )
    reconstr = info['reconstructed']

    # face right, reconstructed
    ax[2, i].imshow(reconstr[0].mu.data.squeeze(), cmap='gray')

    no_ticks(ax[0, i])
    no_ticks(ax[1, i])
    no_ticks(ax[2, i])

  return fig

def viz_reconstruction_all_views(experiment, dataset, indices):
  num_examples = len(indices)
  fig, ax = plt.subplots(num_groups, num_examples, figsize=(12, 12))

  for i, subj in enumerate(indices):
    batch = dataset[subj]
    group_mask = batch[0]
    views = batch[1:]

    inference_group_mask = torch.ones(num_groups, num_groups) - torch.eye(num_groups)
    views = [x.unsqueeze(0) for x in views]
    if experiment.use_gpu:
      inference_group_mask = inference_group_mask.cuda()
      views = [x.cuda() for x in views]

    info = experiment.vae.reconstruct(
      [Variable(x) for x in views],
      Variable(inference_group_mask)
    )
    reconstr = info['reconstructed']

    for j in range(num_groups):
      ax[j, i].imshow(reconstr[j].mu.data[j].squeeze(), cmap='gray')
      no_ticks(ax[j, i])

  plt.tight_layout()
  return fig

# def viz_missing_training_view_reconstruction(experiment):
#   fig, ax = plt.subplots(4, num_groups, figsize=(12, 12))

#   # 1 is normal, the other three have are missing the last group
#   indices = [1, 35, 44, 93]

#   path = Path('data/cvl_faces/')

#   # all_subjects = range(1, 114 + 1)
#   train_subjects = range(1, 100 + 1)
#   test_subjects = range(101, 114 + 1)

#   self.train_dataset = CVLDataset(path, subjects=train_subjects, transform=transform)
#   self.train_loader = torch.utils.data.DataLoader(
#     self.train_dataset,
#     batch_size=self.batch_size,
#     shuffle=True
#   )

#   self.test_dataset = CVLDataset(path, subjects=test_subjects, transform=transform)
#   self.test_loader = torch.utils.data.DataLoader(
#     self.test_dataset,
#     batch_size=self.batch_size,
#     shuffle=True
#   )

#   for i, subj in enumerate(indices):
#     batch = experiment.test_dataset[subj]
#     group_mask = batch[0]
#     views = batch[1:]

#     inference_group_mask = torch.ones(num_groups, num_groups) - torch.eye(num_groups)
#     views = [x.unsqueeze(0) for x in views]
#     if experiment.use_gpu:
#       inference_group_mask = inference_group_mask.cuda()
#       views = [x.cuda() for x in views]

#     info = experiment.vae.reconstruct(
#       [Variable(x) for x in views],
#       Variable(inference_group_mask)
#     )
#     reconstr = info['reconstructed']

#     for j in range(num_groups):
#       ax[j, i].imshow(reconstr[j].mu.data[j].squeeze(), cmap='gray')
#       no_ticks(ax[j, i])

#   plt.tight_layout()
#   return fig

if __name__ == '__main__':
  # dataset = CVLDataset(Path('data/cvl_faces/'), transform=transform)
  # dataloader = torch.utils.data.DataLoader(
  #   dataset,
  #   batch_size=5,
  #   shuffle=True
  # )

  experiment = CVLFacebackExperiment(
    dim_z=128,
    batch_size=16,
    lam=0.1,
    sparsity_matrix_lr=1e-4,
    inference_net_output_dim=128,
    generative_net_input_dim=128,
    initial_baseline_precision=100,
    prior_theta_sigma=1,
    group_available_prob=0.9,
    inference_net_num_filters=128,
    generative_net_num_filters=128,
    use_gpu=True,
    base_results_dir=Path('results/'),
    prefix='deleteme_faces_'
  )
  experiment.train(1000)
