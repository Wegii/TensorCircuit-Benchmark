# TensorCircuit Benchmark
This repository contains the code for benchmarking the Tensorcircuit library on a standard machine learning problem.
The goal is to show the performance of TensorCircuit over similar libraries such as Qiskit or Pennylane.
The utilized task is based on the simple MNIST classification task.

The codebase consists of two distinct implementations:
- TensorCircuit and JAX implementation for MNIST classification
- Pennylane and Pytorch implementation for MNIST classification

### Running the code 
To run the default simulations, execute:
```python
python3 testbed.py
```

For benchmarking the Parametrized Quantum Circuits, execute:
```python
LINE_PROFILE=1 python3 testbed.py
```

It is possible to select the utilized library (either tensorcircuit or pennylane) as well as the utilized device to run
on. For pennylane, only CPU is available. See testbed.py for more information.

### Requirements
The implementation requires many different libraries. To run either the TensorCircuit or Pennylane simulations, installing these libraries in distinct environments is recommended, as not all the necessary libraries are compatible.

The code was tested on a GPU system, utilizing **CUDA12.x** and **CUDNN8.x**. TensorCircuit, JAX, Pennylane, Cirq were
installed using pip.