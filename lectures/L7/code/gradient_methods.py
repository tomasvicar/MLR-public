"""Compare three gradients for a single sigmoid neuron, with z = 1.

    MPLBACKEND=Agg uv run python lectures/L7/code/gradient_methods.py
"""

import numpy as np
import torch

INPUTS = np.array([1.0, 2.0])
TARGET = 1.0


def objective(weights):
    """Squared-error loss, with the inputs and target kept fixed."""
    return 0.5 * (1 / (1 + np.exp(-(weights @ INPUTS))) - TARGET) ** 2


def main():
    weights = np.array([0.5, 0.25])
    z = weights @ INPUTS
    sigmoid = 1 / (1 + np.exp(-z))
    analytical = (sigmoid - TARGET) * sigmoid * (1 - sigmoid) * INPUTS
    epsilon = 1e-5
    numerical = np.array([
        (objective(weights + epsilon * direction)
         - objective(weights - epsilon * direction)) / (2 * epsilon)
        for direction in np.eye(weights.size)
    ])

    w = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(INPUTS, dtype=torch.float64)
    potential = w @ x
    activation = torch.sigmoid(potential)
    residual = activation - TARGET
    loss = 0.5 * residual**2
    loss.backward()
    automatic = w.grad.detach().numpy()

    print(f"Pre-activation: {z:.6f}")
    print(f"Activation:    {sigmoid:.8f}")
    print(f"Loss:          {objective(weights):.8f}")
    print(f"Analytical:    {analytical}")
    print(f"Numerical:     {numerical} (epsilon = {epsilon:g})")
    print(f"Autograd:      {automatic}")
    print(f"Max finite-difference error: {np.max(np.abs(numerical - analytical)):.2e}")
    assert z == 1.0
    np.testing.assert_allclose(analytical, [-0.052877092784266715, -0.10575418556853343],
                               rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(automatic, analytical, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    main()
