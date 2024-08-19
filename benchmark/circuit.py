import tensorcircuit as tc
import pennylane as qml
import torch
from line_profiler import profile

@profile
def qml_ys_tc(x, weights, nlayers):
    n = 9
    weights = tc.backend.cast(weights, "complex128")
    x = tc.backend.cast(x, "complex128")
    c = tc.Circuit(n)

    for i in range(n):
        c.rx(i, theta=x[i])

    for j in range(nlayers):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=weights[2 * j, i])
            c.ry(i, theta=weights[2 * j + 1, i])

    ypreds = []
    for i in range(n):
        ypred = c.expectation([tc.gates.z(), (i,)])
        ypred = tc.backend.real(ypred)
        ypred = (tc.backend.real(ypred) + 1) / 2.0
        ypreds.append(ypred)

    return tc.backend.stack(ypreds)


@profile
def qml_ys_pl(x: torch.tensor, weights: torch.nn.Parameter, n_qubits: int = 9, n_layers: int = 3):
    dev = qml.device('cirq.simulator', wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(x: torch.tensor, weights: torch.nn.Parameter, n_qubits: int, n_layers: int):
        for i in range(n_qubits):
            qml.RX(x[i], i)
    
        for j in range(n_layers):
            for i in range(n_qubits - 1):
                qml.CNOT([i, i + 1])
            for i in range(n_qubits):
                qml.RX(weights[2 * j, i], i)
                qml.RY(weights[2 * j + 1, i], i)
    
        ypreds = []
        for i in range(n_qubits):
            ypred = qml.expval(qml.PauliZ(i))
            ypreds.append(ypred)
            
        return ypreds
        
    ret = circuit(x = x,
                  weights = weights,
                  n_qubits = n_qubits,
                  n_layers = n_layers)
    
    # Normalize to [0, 1]
    for i in range(n_qubits):
        ret[i] = (torch.real(ret[i]) + 1)/2.0

    return torch.stack(ret, dim=0)