import torch
import numpy as np
import hypothesis
import matplotlib.pyplot as plt
import corner
import os
from scipy.stats import gaussian_kde

from torch.distributions.normal import Normal

from hypothesis.visualization.mcmc import plot_density
from hypothesis.transition import Normal as NormalTransition
from hypothesis.inference.mcmc import ClassifierMetropolisHastings

truth = 0
epoch = 99
regularization = "wo"
path = "results/lfi_{}_regularization_{}.th".format(regularization, epoch)

if not os.path.exists(path):

    classifier = torch.load('models/{}_regularization_{}'.format(regularization, epoch), map_location='cpu')
    min_bound = torch.FloatTensor([-10])
    max_bound = torch.FloatTensor([10])
    classifier.lower = min_bound
    classifier.upper = max_bound
    classifier.eval()

    N = Normal(truth, 1)
    observations = N.sample(torch.Size([10, 1]))
    transition = NormalTransition(0.05)

    sampler = ClassifierMetropolisHastings(classifier, transition)
    theta_0 = torch.FloatTensor([5])
    result = sampler.infer(
        observations,
        theta_0=theta_0,
        samples=1000000,
        burnin_samples=100000)

    hypothesis.save(result, path)

else:
    result = hypothesis.load(path)

figure = corner.corner(
   result.get_chain(0).numpy(),
   show_titles=True,
   bins=50,
   top_ticks=False,
   label_kwargs=dict(fontsize=18),
   title_kwargs=dict(fontsize=18))

plt.savefig("plots/{}_regularization_{}.pdf".format(regularization, epoch), bbox_inches="tight", pad_inches=0)
