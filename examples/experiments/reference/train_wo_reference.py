import pickle
import argparse
import torch
import numpy as np
import os
import re

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


def main(arguments):
    # Data-source preperation.
    simulation_dataset = SimulationDataset()
    # Training preperation.
    real = torch.ones(arguments.batch_size, 1).cuda()
    fake = torch.zeros(arguments.batch_size, 1).cuda()
    bce = torch.nn.BCELoss().cuda()
    iterations = int(arguments.iterations / arguments.batch_size)

    classifier = allocate_classifier(arguments.hidden).cuda()
    optimizer = torch.optim.Adam(classifier.parameters())

    for epoch in range(arguments.epochs):
        simulation_loader = iter(DataLoader(simulation_dataset, num_workers=0, batch_size=arguments.batch_size))
        for iteration in range(iterations):
            theta1, x_theta1 = next(simulation_loader)
            theta2, x_theta2 = next(simulation_loader)

            theta1, x_theta1, theta2, x_theta2 = theta1.cuda(), x_theta1.cuda(), theta2.cuda(), x_theta2.cuda()

            in_real1 = torch.cat([theta1, x_theta1], dim=1).detach()
            in_fake1 = torch.cat([theta1, x_theta2], dim=1).detach()
            y_real1 = classifier(in_real1)
            y_fake1 = classifier(in_fake1)

            in_real2 = torch.cat([theta2, x_theta2], dim=1).detach()
            in_fake2 = torch.cat([theta2, x_theta1], dim=1).detach()
            y_real2 = classifier(in_real2)
            y_fake2 = classifier(in_fake2)

            loss = bce(y_real1, real) + bce(y_fake1, fake) + bce(y_real2, real) + bce(y_fake2, fake)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(classifier, "models_" + str(arguments.run) + "/" + str(arguments.hidden) + '_' + str(epoch) + ".th")
    torch.save(classifier, "models_" + str(arguments.run) + "/" + str(arguments.hidden) + "_final.th")


def allocate_classifier(hidden):
    classifier = torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Sigmoid())

    return classifier

class SimulationDataset(Dataset):

    def __init__(self):
        super(SimulationDataset, self).__init__()

        with open("presimulated_datasets/simulation_dataset_long.pickle", "rb") as presimulated_file:
            self.data = pickle.load(presimulated_file)

        self.size = self.data.size(0)

    def __getitem__(self, index):
        row = self.data[index]
        theta, x = row[:1], row[1:]

        return theta, x

    def __len__(self):
        return self.size


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free MCMC. Training.")
    parser.add_argument("--num-thetas", type=int, default=1000, help="Number of thetas to generate.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for every theta.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations within a single epoch.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch-size")
    parser.add_argument("--hidden", type=int, default=500, help="Number of hidden units.")
    parser.add_argument("--run", type=int, default=1, help="Experiment run.")
    arguments, _ = parser.parse_known_args()
    if arguments.iterations is None:
        arguments.iterations = arguments.num_thetas * arguments.num_samples

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
