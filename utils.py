# import itertools
# import random
import torch

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
