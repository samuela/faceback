# import itertools
# import random
import torch
import numpy as np
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

def viz_sparsity(vae, group_names=None):
  """Visualize the sparisty matrix associating latent components with groups.
  Columns are normalized to have norm 1."""
  fig, ax = plt.subplots()
  mat = vae.sparsity_matrix().data.cpu()
  ax.imshow((mat / torch.max(mat, dim=0)[0]).numpy())
  ax.xaxis.tick_top()

  ax.xaxis.set_label_position('top')
  ax.set_xlabel('latent components', fontsize=20)
  ax.set_ylabel('group generative nets', fontsize=20)

  ax.xaxis.set_ticks(np.arange(mat.size(1)))
  ax.xaxis.set_ticklabels(np.arange(1, mat.size(1) + 1))

  ax.yaxis.set_ticks(np.arange(mat.size(0)))
  if group_names is not None:
    ax.yaxis.set_ticklabels(group_names)
  else:
    ax.yaxis.set_ticklabels(np.arange(1, mat.size(0) + 1))

  plt.tight_layout()

  # plt.colorbar()
  return fig, ax
