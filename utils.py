# import itertools
# import random
import torch
import matplotlib.pyplot as plt

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

# def sample_random_mask_slow(rows, cols):
#   """This is slow but guarantees that each row has at least one entry."""
#   combos = list(itertools.product([0, 1], repeat=cols))[1:]
#   return torch.FloatTensor([random.choice(combos) for _ in range(rows)])

def sample_random_mask(rows, cols, prob):
  return (torch.rand(rows, cols) < prob).float()

def viz_sparsity(vae):
  """Visualize the sparisty matrix associating latent components with groups.
  Columns are normalized to have norm 1."""
  fig, ax = plt.subplots()
  mat = vae.sparsity_matrix().data
  ax.imshow((mat / torch.max(mat, dim=0)[0]).numpy())
  ax.xaxis.tick_top()
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.set_label_position('top')
  # plt.colorbar()
  return fig, ax
