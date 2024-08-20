import numpy as np
import tensorflow as tf
import torch
torch.set_default_dtype(torch.float64)
import numpy as np

from utils import load_mnist_pt
from circuit import qml_ys_pl

from line_profiler import profile


class qNet(torch.nn.Module):
    """ Quantum Neural Network consisting of a Quantum Circuit """

    def __init__(self, n_qubits: int = 9, n_layers: int = 3):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Weights for quantum circuit
        self.q_weights = torch.nn.Parameter(torch.randn([2 * n_layers, n_qubits]))

    def forward(self, inputs):
        return qml_ys_pl(inputs,
                      self.q_weights,
                      n_qubits = self.n_qubits,
                      n_layers = self.n_layers)


def pl_pt_benchmark(batch_size: int = 32, n_qubits: int = 9, n_layers: int = 3):
    """ Start the benchmark for the quantum circuit and neural network with pytorch and pennylane

    :param batch_size: Batch size of dataset
    :param n_qubits: Number of qubits in the Circuit
    :param n_layers: Number of layers in the Circuit
    :return: None
    """

    mnist_data = load_mnist_pt(batch_size)

    # Construct circuit and network
    qnetwork = qNet(n_qubits=n_qubits,
                    n_layers=n_layers)
    qmodel = torch.nn.Sequential(qnetwork,
                                 torch.nn.Linear(9, 1),
                                 torch.nn.Sigmoid())
    
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(qmodel.parameters(), lr = 1e-3)
    optimizer.zero_grad()
    
    with torch.set_grad_enabled(True):
        for i, (xs, ys) in enumerate(mnist_data):
            # Iterate over batch
            yps = []
            for j in range(batch_size):
                yp = qmodel(xs[j])
                yps.append(yp)


            yps = torch.stack(yps)

            # Calculate loss over batch
            loss_ret = loss(
                torch.reshape(yps, [batch_size, 1]), torch.reshape(ys, [batch_size, 1])
            )
            
            loss_ret.backward()
            optimizer.step()

            if i%5 == 0:
                print(f"Executed {i}/{len(mnist_data)}")
        
            print(loss)

