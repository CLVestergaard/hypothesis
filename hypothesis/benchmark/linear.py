import numpy as np
import torch



def allocate_observations(theta, observations=100):
    inputs = torch.tensor(theta).float().repeat(observations).view(-1, 3)
    outputs = simulator(inputs)

    return outputs


def simulator(inputs):
    with torch.no_grad():
        inputs = inputs.view(-1, 3)
        samples = inputs.size(0)
        m = inputs[:, 0].numpy()
        b = inputs[:, 1].numpy()
        f = inputs[:, 2].numpy()
        x = 10 * np.random.rand(samples)
        y_err = .1 + .5 * np.random.rand(samples)
        y = m * x + b
        y += np.abs(f * y) * np.random.randn(samples)
        y += y_err * np.random.randn(samples)
        x = torch.tensor(x).view(-1, 1).float()
        y = torch.tensor(y).view(-1, 1).float()
        y_err = torch.tensor(y_err).view(-1, 1).float()
        outputs = torch.cat([x, y, y_err], dim=1)

    return outputs
