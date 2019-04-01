import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import hypothesis

from torch.distributions.normal import Normal



def main():

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,2.5))
    ax1 = axes[0]
    ax1.set_ylim(0, 1.1)
    ax1.minorticks_on()
    ax2 = axes[1]
    ax2.minorticks_on()
    ax2.set_ylim(0, 1.1)

    for ref in ["w", "wo"]:

        classifiers = load_classifiers(ref)
        observations = get_observations()

        min_x = np.amin(0)
        max_x = np.amax(5)
        x_range = np.linspace(float(min_x), float(max_x), 1000)

        #y_min = []
        #_max = []
        #y_means = []

        #also get the theta, which produces the largest gap
        #max_gap = 0
        #max_gap_theta = None
        #for x in x_range:
        #    tmp_y_min, tmp_y_max = None, None
        #    theta = torch.Tensor([[x]] * 10)
        #    input = torch.cat([theta, observations], dim=1).detach()
        #    for classifier in classifiers:
        #        s = classifier(input).log().sum().exp().item()
        #        if tmp_y_min is None or tmp_y_min > s:
        #            tmp_y_min = s
        #        if tmp_y_max is None or tmp_y_max < s:
        #            tmp_y_max = s
        #    if tmp_y_max - tmp_y_min > max_gap:
        #        max_gap = tmp_y_max - tmp_y_min
        #        max_gap_theta = x
        #    y_min.append(tmp_y_min)
        #    y_max.append(tmp_y_max)

        #then get classifiers, which produce min / max output
        #print("{} reference: maximum gap ({}) produced at theta {}.".format(ref, max_gap, max_gap_theta))
        #min_output, max_output = None, None
        #min_classifier_index, max_classifier_index = None, None

        #theta = torch.Tensor([[max_gap_theta]] * 10)
        #input = torch.cat([theta, observations], dim=1).detach()

        #for i in range(len(classifiers)):
        #    s = classifiers[i](input).log().sum().exp().item()
        #    if min_output is None or min_output > s:
        #        min_output = s
        #        min_classifier_index = i
        #    if max_output is None or max_output < s:
        #        max_output = s
        #        max_classifier_index =i
        #print("classifier indexes: {} and {}".format(min_classifier_index, max_classifier_index))
        #print("##############################")

        # variance version
        y_min = []
        y_max = []
        y_means = []
        for x in x_range:
            ys = []
            theta = torch.Tensor([[x]] * 10)
            input = torch.cat([theta, observations], dim=1).detach()
            for classifier in classifiers:
                s = classifier(input).log().sum().exp().item()
                ys.append(s)
            y_mean = np.mean(ys)
            y_means.append(y_mean)
            y_std = np.std(ys)
            y_min.append(y_mean-y_std)
            y_max.append(y_mean+y_std)


        if ref == "w":
            ax1.fill_between(x_range, y_min, y_max, color='black', alpha=.25)
            ax1.set_ylim(bottom=0)
            ax1.plot(x_range, y_means, color="black")
            ax1.axvline(5, c='C0', lw=2, linestyle='-', alpha=.95)
            ax1.set_ylabel(r'$s(x, \theta)$')
            ax1.set_xlabel(r'$\theta$')

            ax3 = ax1.twinx()
            ax3.minorticks_on()
            data = hypothesis.load('results/lf-w-reference-classifier').chain.tolist()
            data = [x[0] for x in data]
            weights = [1./len(data)] * len(data)
            ax3.hist(data, histtype="stepfilled", color="black", density=False, alpha=.25, weights=weights)
            ax3.set_yticks([0, 0.05, 0.1])
            #ax3.set_ylabel(r'$p(\theta|x)$', color='C0')
            #ax3.tick_params('y', which='both', colors='C0')


        else:
            ax2.fill_between(x_range, y_min, y_max, color='black', alpha=.25)
            ax2.set_ylim(bottom=0)
            ax2.plot(x_range, y_means, color="black")
            ax2.axvline(5, c='C0', lw=2, linestyle='-', alpha=.95)
            #ax2.set_ylabel(r'$s(x, \theta)$')
            ax2.set_xlabel(r'$\theta$')

            ax4 = ax2.twinx()
            ax4.minorticks_on()
            data = hypothesis.load('results/lf-wo-reference-classifier').chain.tolist()
            data = [x[0] for x in data]
            weights = [1./len(data)] * len(data)
            ax4.hist(data, histtype="stepfilled", color="black", density=False, alpha=.25, weights=weights)
            ax4.set_ylabel(r'$p(\theta|x)$')
            #ax4.tick_params('y', which='both', colors='C0')

    plt.tight_layout()
    plt.savefig("plots/decision.pdf".format(ref))

def load_classifiers(ref):
    classifiers = []
    for i in range(1, 11):
        classifier = torch.load("models_{}_reference/run{}_500_final.th".format(ref, i), map_location='cpu')
        classifier.eval()
        classifiers.append(classifier)
    return classifiers

def get_observations():
    path = "observations/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "observations.th"
    if not os.path.exists(path):
        N = Normal(5, 1.)
        observations = N.sample(torch.Size([10, 1]))
        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations

if __name__ == "__main__":
    main()
