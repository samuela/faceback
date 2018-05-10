import itertools

import torch
from torch.autograd import Variable
# from kindling.nan_police import torch

from kindling.distributions import Normal
from kindling.utils import KL_Normals, KL_Normals_independent


class FacebackVAE(object):
  """An implementation of a Faceback variational autoenoder. Note that this
  particular implementation requires that the prior over z and the approximate
  posterior q(z | x) must both be diagonal normal distributions.

  This is the mixture of Gaussians model.
  """

  def __init__(
      self,
      inference_nets,
      generative_nets,
      mixture_weight_net,
      prior_z
  ):
    """Set things up.

    Arguments
    =========
    inference_nets : a list of conditional distributions returning diagonal
      Gaussian approximate posteriors over z for each of the groups.
    generative_nets : a list of conditional distributions, p(x_i | z)
    mixture_weight_net : a network taking in Xs in list form and returning the
      pre-softmax mixture weights.
    prior_z : distribution
      the prior over z.
    """
    self.inference_nets = inference_nets
    self.generative_nets = generative_nets
    self.mixture_weight_net = mixture_weight_net
    self.prior_z = prior_z

  def elbo(self, Xs, group_mask, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    group_mask : binary Variable of shape [batch_size, num_groups]
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    scalar Variable with the ELBO
    """
    batch_size = Xs[0].size(0)
    num_groups = len(Xs)

    def sub_elbo(ix):
      """Calculate the ELBO using just group ix for inference. Returns a
      Variable of shape [batch_size]."""
      q_z = self.inference_nets[ix](Xs[ix])

      # KL divergence is additive across independent joint distributions, so
      # this works appropriately. This is where the normal constraints come
      # from. shape [batch_size].
      z_kl = torch.sum(
        KL_Normals_independent(q_z, self.prior_z.expand_as(q_z)),
        dim=1
      )

      # [batch_size, dim_z]
      z_sample = q_z.sample()

      # Evaluating the log likelihood always happens on all available data. Sum
      # the likelihoods over all of the available groups.
      loglik_term = sum(
        # Sum the likelihoods over every observed dimension in a group. Then
        # mask based on whether or not this group is present in each batch
        # item.
        torch.sum(g(z_sample).logprob_independent(X), dim=1) * group_mask[:, ix]
        for ix, (g, X) in enumerate(zip(self.generative_nets, Xs))
      )

      return -z_kl + loglik_term

    # [batch_size, num_groups]
    sub_elbos = torch.stack([sub_elbo(ix) for ix in range(num_groups)], dim=1)
    mix_weights = self.mixture_weights(Xs, inference_group_mask)
    elbo = torch.sum(mix_weights * sub_elbos) / batch_size

    return {
      'elbo': elbo,
      'mixture_weights': mix_weights,
      'sub_elbos': sub_elbos
    }

  def mixture_weights(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    Variable of shape [batch_size, num_groups] with the mixture weights for each
    group.
    """
    # Calculate mixture weights but mask based on which groups we actually want
    # to run inference on.
    mixture_weights_unnorm = (
      torch.exp(self.mixture_weight_net(Xs, inference_group_mask)) * inference_group_mask
    )
    mixture_weight_norms = torch.sum(mixture_weights_unnorm, dim=1)
    return mixture_weights_unnorm / mixture_weight_norms.view(-1, 1)

  def reconstruct(self, Xs, inference_group_mask):
    """Don't try to differentiate through this! The multinomial will produce
    high-variance gradients.

    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    """
    mix_weights = self.mixture_weights(Xs, inference_group_mask)
    mix_sample = torch.multinomial(mix_weights, 1)
    q_z_sample = torch.stack([
      self.inference_nets[ix](Xs[ix]).sample()
      for ix in mix_sample.data.view(-1)
    ])
    return {
      'mixture_weights': mix_weights,
      'mixture_sample': mix_sample,
      'reconstructed': [gen(q_z_sample) for gen in self.generative_nets]
    }

  def parameters(self):
    return itertools.chain(
      *[net.parameters() for net in self.inference_nets],
      *[net.parameters() for net in self.generative_nets],
      self.mixture_weight_net.parameters()
    )

class FacebackAveragingVAE(object):
  """This currently assumes that all of the inference net posteriors are Normal
  distributions."""

  def __init__(
      self,
      inference_nets,
      generative_nets,
      prior_z
  ):
    self.inference_nets = inference_nets
    self.generative_nets = generative_nets
    self.prior_z = prior_z

  def elbo(self, Xs, group_mask, inference_group_mask, mc_samples=1):
    batch_size = Xs[0].size(0)
    # num_groups = len(Xs)

    # [batch_size, dim_z]
    q_z = self._q_z(Xs, inference_group_mask)

    # KL divergence is additive across independent joint distributions, so this
    # works appropriately. This is where the normal constraints come from.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)

    # List of [batch_size * mc_samples, dim_group_x]
    Xsrep = [Variable(X.data.repeat(mc_samples, 1)) for X in Xs]

    # Evaluating the log likelihood always happens on all available data. Sum
    # the likelihoods over all of the available groups.
    loglik_term = sum(
      # Sum the likelihoods over every observed dimension in a group. Then
      # mask based on whether or not this group is present in each batch
      # item.
      torch.sum(
        torch.sum(g(z_sample).logprob_independent(X), dim=1) * group_mask[:, ix]
      )
      for ix, (g, X) in enumerate(zip(self.generative_nets, Xsrep))
    ) / mc_samples / batch_size

    elbo = -z_kl + loglik_term

    return {
      'elbo': elbo,
      'z_kl': z_kl,
      'reconstruction_log_likelihood': loglik_term,
      'z_sample': z_sample
    }

  def reconstruct(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    """
    q_z = self._q_z(Xs, inference_group_mask)
    z_sample = q_z.sample()

    return {
      'q_z': q_z,
      'z_sample': z_sample,
      'reconstructed': [gen(z_sample) for gen in self.generative_nets]
    }

  def _q_z(self, Xs, inference_group_mask):
    q_zs = [q(X) for q, X in zip(self.inference_nets, Xs)]
    num_active_inference = torch.sum(inference_group_mask, dim=1)
    avg_mu = sum(
      q_z.mu * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    ) / num_active_inference.view(-1, 1)
    avg_sigma = sum(
      q_z.sigma * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    ) / num_active_inference.view(-1, 1)

    return Normal(avg_mu, avg_sigma)

  def parameters(self):
    return itertools.chain(
      *[net.parameters() for net in self.inference_nets],
      *[net.parameters() for net in self.generative_nets]
    )

class SparseProductOfExpertsVAE(object):
  """A product of experts (PoE) style posterior approximation with sparsity."""
  def __init__(
      self,
      inference_nets,
      generative_nets,
      prior_z,
      prior_theta,
      lam
  ):
    """
    Arguments
    =========
    inference_nets : list of Normal_MeanPrecisionNet
    generative_nets : list of DistributionNets
    prior_z : Normal distribution of shape [1, dim_z]
    """
    self.inference_nets = inference_nets
    self.generative_nets = generative_nets
    self.prior_z = prior_z
    self.prior_theta = prior_theta
    self.lam = lam

    self.num_groups = len(self.generative_nets)
    self.dim_z = self.prior_z.size(1)

    self.sparsity_matrix = Variable(
      # (1.0 / math.sqrt(self.dim_z)) * torch.randn(self.num_groups, self.dim_z),
      # torch.ones(self.num_groups, self.dim_z),
      torch.eye(2), #### XXX
      requires_grad=True
    )

  def elbo(self, Xs, group_mask, inference_group_mask, mc_samples=1):
    batch_size = Xs[0].size(0)

    # [batch_size, dim_z]
    q_z = self.approx_posterior(Xs, inference_group_mask)

    # KL divergence is additive across independent joint distributions, so this
    # works appropriately. This is where the normal constraints come from.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
    loglik_term = self.log_likelihood(Xs, group_mask, z_sample, mc_samples)

    logprob_theta = sum(self.prior_theta.logprob(net) for net in self.generative_nets)
    logprob_L1 = -self.lam * torch.sum(torch.abs(self.sparsity_matrix))

    # `loss` is the differentiable part of the ELBO. We negate it in order to do
    # descent. `elbo` is the complete ELBO including the L1 sparsity prior.
    loss = -1 * (-z_kl + loglik_term + logprob_theta)
    elbo = -loss + logprob_L1

    return {
      'logprob_theta': logprob_theta,
      'logprob_L1': logprob_L1,
      'loss': loss,
      'elbo': elbo,
      'z_kl': z_kl,
      'reconstruction_log_likelihood': loglik_term,
      'q_z': q_z,
      'z_sample': z_sample
    }

  def approx_posterior(self, Xs, inference_group_mask):
    """Run the inference net to evaluate q(z | x), the approximate posterior."""
    q_zs = [q(X) for q, X in zip(self.inference_nets, Xs)]

    # Here we also have to add in the precision from the prior. The sparsity
    # matrix mask is squared in order to prevent any negative values.
    precision = self.prior_z.sigma.pow(-2) + sum(
      self.sparsity_matrix[i].pow(2) * q_z.precision * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    )

    mu = precision.pow(-1) * sum(
      q_z.precision * q_z.mu * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    )

    return Normal(mu, precision.pow(-0.5))

  def log_likelihood(self, Xs, group_mask, z_sample, mc_samples=1):
    batch_size = Xs[0].size(0)

    # List of [batch_size * mc_samples, dim_group_x]
    Xsrep = [Variable(X.data.repeat(mc_samples, 1)) for X in Xs]

    # Evaluating the log likelihood always happens on all available data. Sum
    # the likelihoods over all of the available groups.
    return sum(
      # Sum the likelihoods over every observed dimension in a group. Then
      # mask based on whether or not this group is present in each batch
      # item.
      torch.sum(
        # Multiply the z_sample with the corresponding sparsity diagonal matrix.
        torch.sum(g(z_sample * self.sparsity_matrix[ix].view(1, -1)).logprob_independent(X), dim=1) * group_mask[:, ix]
      )
      for ix, (g, X) in enumerate(zip(self.generative_nets, Xsrep))
    ) / mc_samples / batch_size

  def proximal_step(self, t):
    norms = torch.abs(self.sparsity_matrix.data)
    self.sparsity_matrix.data.sign_()
    self.sparsity_matrix.data.mul_(torch.clamp(norms - t, min=0))

  def reconstruct(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    """
    q_z = self.approx_posterior(Xs, inference_group_mask)
    z_sample = q_z.sample()

    return {
      'q_z': q_z,
      'z_sample': z_sample,
      'reconstructed': [
        gen(z_sample * self.sparsity_matrix[ix].view(1, -1))
        for ix, gen in enumerate(self.generative_nets)
      ]
    }

  # def parameters(self):
  #   return itertools.chain(
  #     *[net.parameters() for net in self.inference_nets],
  #     *[net.parameters() for net in self.generative_nets],
  #     [self.sparsity_matrix]
  #   )

class OneSidedFacebackoiVAE(object):
  """A product of experts (PoE) style posterior approximation but with sparsity
  only on the generative side."""
  def __init__(
      self,
      inference_nets,
      generative_nets,
      prior_z,
      prior_theta,
      lam
  ):
    """
    Arguments
    =========
    inference_nets : list of Normal_MeanPrecisionNet
    generative_nets : list of DistributionNets
    prior_z : Normal distribution of shape [1, dim_z]
    """
    self.inference_nets = inference_nets
    self.generative_nets = generative_nets
    self.prior_z = prior_z
    self.prior_theta = prior_theta
    self.lam = lam

    self.num_groups = len(self.generative_nets)
    self.dim_z = self.prior_z.size(1)

    self.sparsity_matrix = Variable(
      # (1.0 / math.sqrt(self.dim_z)) * torch.randn(self.num_groups, self.dim_z),
      torch.ones(self.num_groups, self.dim_z),
      requires_grad=True
    )

  def elbo(self, Xs, group_mask, inference_group_mask, mc_samples=1):
    batch_size = Xs[0].size(0)

    # [batch_size, dim_z]
    q_z = self.approx_posterior(Xs, inference_group_mask)

    # KL divergence is additive across independent joint distributions, so this
    # works appropriately. This is where the normal constraints come from.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
    loglik_term = self.log_likelihood(Xs, group_mask, z_sample, mc_samples)

    logprob_theta = sum(self.prior_theta.logprob(net) for net in self.generative_nets)
    logprob_L1 = -self.lam * torch.sum(torch.abs(self.sparsity_matrix))

    # `loss` is the differentiable part of the ELBO. We negate it in order to do
    # descent. `elbo` is the complete ELBO including the L1 sparsity prior.
    loss = -1 * (-z_kl + loglik_term + logprob_theta)
    elbo = -loss + logprob_L1

    return {
      'logprob_theta': logprob_theta,
      'logprob_L1': logprob_L1,
      'loss': loss,
      'elbo': elbo,
      'z_kl': z_kl,
      'reconstruction_log_likelihood': loglik_term,
      'q_z': q_z,
      'z_sample': z_sample
    }

  def approx_posterior(self, Xs, inference_group_mask):
    """Run the inference net to evaluate q(z | x), the approximate posterior."""
    q_zs = [q(X) for q, X in zip(self.inference_nets, Xs)]

    # Here we also have to add in the precision from the prior. The sparsity
    # matrix mask is squared in order to prevent any negative values.
    precision = self.prior_z.sigma.pow(-2) + sum(
      q_z.precision * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    )

    mu = precision.pow(-1) * sum(
      q_z.precision * q_z.mu * inference_group_mask[:, i].contiguous().view(-1, 1)
      for i, q_z in enumerate(q_zs)
    )

    return Normal(mu, precision.pow(-0.5))

  def log_likelihood(self, Xs, group_mask, z_sample, mc_samples=1):
    batch_size = Xs[0].size(0)

    # List of [batch_size * mc_samples, dim_group_x]
    Xsrep = [Variable(X.data.repeat(mc_samples, 1)) for X in Xs]

    # Evaluating the log likelihood always happens on all available data. Sum
    # the likelihoods over all of the available groups.
    return sum(
      # Sum the likelihoods over every observed dimension in a group. Then
      # mask based on whether or not this group is present in each batch
      # item.
      torch.sum(
        # Multiply the z_sample with the corresponding sparsity diagonal matrix.
        torch.sum(g(z_sample * self.sparsity_matrix[ix].view(1, -1)).logprob_independent(X), dim=1) * group_mask[:, ix]
      )
      for ix, (g, X) in enumerate(zip(self.generative_nets, Xsrep))
    ) / mc_samples / batch_size

  def proximal_step(self, t):
    norms = torch.abs(self.sparsity_matrix.data)
    self.sparsity_matrix.data.sign_()
    self.sparsity_matrix.data.mul_(torch.clamp(norms - t, min=0))

  def reconstruct(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    """
    q_z = self.approx_posterior(Xs, inference_group_mask)
    z_sample = q_z.sample()

    return {
      'q_z': q_z,
      'z_sample': z_sample,
      'reconstructed': [
        gen(z_sample * self.sparsity_matrix[ix].view(1, -1))
        for ix, gen in enumerate(self.generative_nets)
      ]
    }

  # def parameters(self):
  #   return itertools.chain(
  #     *[net.parameters() for net in self.inference_nets],
  #     *[net.parameters() for net in self.generative_nets],
  #     [self.sparsity_matrix]
  #   )

class VotingMixtureWeightNet(object):
  def __init__(self, embeddings, W, b):
    """Each group votes independently for its mixture weight. Group input data
    is mapped into a shared embedding space and then linearly weighted.

    Arguments
    =========
    embeddings : list of networks mapping the group inputs to a shared embedding
    W : Variable of shape [num_groups, dim_shared]
    b : Variable of shape [num_groups]
    """
    self.embeddings = embeddings
    self.W = W
    self.b = b

  def __call__(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    Variable of shape [batch_size, num_groups].
    """
    # return torch.cat(
    #   [shared_inference_nets[ix](X) for ix, X in enumerate(Xs)],
    #   dim=1
    # ) @ self.W

    unclamped = torch.stack(
      [self.embeddings[ix](X) @ self.W[ix] + self.b[ix]
       for ix, X in enumerate(Xs)],
      dim=1
    )

    # Clamp these values since they will be exp-ed down the road.
    return torch.clamp(unclamped, -5, 5)

  def parameters(self):
    return [self.W, self.b]

class UniformMixtureWeightNet(object):
  def __call__(self, Xs, inference_group_mask):
    """
    Arguments
    =========
    Xs : list of Variables of shape [batch_size, dim_x] where dim_x can vary.
    inference_group_mask : binary Variable of shape [batch_size, num_groups]

    Returns
    =======
    Variable of shape [batch_size, num_groups].
    """
    batch_size = Xs[0].size(0)
    num_groups = len(Xs)
    return Variable(torch.zeros(batch_size, num_groups))

  def parameters(self):
    return []
