import tensorcircuit as tc
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax
import time
from circuit import qml_ys_tc
from utils import load_mnist_tf
from line_profiler import profile


@profile
def qml_hybrid_loss(x, y, params, nlayers):
    weights = params["qweights"]
    w = params["cweights:w"]
    b = params["cweights:b"]
    ypred = qml_ys_tc(x, weights, nlayers)
    ypred = tc.backend.reshape(ypred, [-1, 1])
    ypred = w @ ypred + b
    ypred = jax.nn.sigmoid(ypred)
    ypred = ypred[0]

    # BCE loss
    loss = -y * tc.backend.log(ypred) - (1 - y) * tc.backend.log(1 - ypred)
    return loss


def training_loop_timed(dataset, params, nlayers, optimizer, opt_state, loss):
    loss_times = []
    opt_times = []
    update_times = []
    count = 0
    
    for i, (xs, ys) in enumerate(dataset):
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        
        t0 = time.time()
        v, grads = loss(xs, ys, params, nlayers)
        t1 = time.time()
        loss_times.append(t1 - t0)

        t0 = time.time()
        updates, opt_state = optimizer.update(grads, opt_state)
        t1 = time.time()
        opt_times.append(t1 - t0)

        t0 = time.time()
        params = optax.apply_updates(params, updates)
        t1 = time.time()
        update_times.append(t1 - t0)
        
        count = count+1

        if i%5 == 0:
            print(f"Executed {i}/{len(dataset)}")

    loss_times = jnp.array(loss_times)
    opt_times = jnp.array(opt_times)
    update_times = jnp.array(update_times)
    
    return ((loss_times, jnp.std(loss_times)), 
            (jnp.mean(opt_times), jnp.std(opt_times)), 
            (jnp.mean(update_times), jnp.std(update_times))
           )
    
def training_loop(dataset, params, nlayers, optimizer, opt_state, loss):
    for i, (xs, ys) in enumerate(dataset):
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        
        t0 = time.time()
        v, grads = loss(xs, ys, params, nlayers)
        t0 = time.time()
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i%5 == 0:
            print(f"Executed {i}/{len(dataset)}")


def tc_jax_benchmark(batch_size: int = 32, n_qubits: int = 9, n_layers: int = 3, profile_path: str = ""):
    # Set backend and precision to double
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    nlayers = n_layers
    
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, num=4)
    params = {
        "qweights": jax.random.normal(subkeys[0], shape=[n_layers * 2, 9]),
        "cweights:w": jax.random.normal(subkeys[1], shape=[9]),
        "cweights:b": jax.random.normal(subkeys[2], shape=[1]),
    }

    qml_hybrid_loss_vag = tc.backend.jit(
        tc.backend.vvag(qml_hybrid_loss, vectorized_argnums=(0, 1), argnums=2),
        static_argnums=3,
    )

    # Set optimizer
    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(params)

    # Load dataset
    mnist_data = load_mnist_tf(batch_size)
    
    t0 = time.time()
    if profile_path:
        with jax.profiler.trace(profile_path, create_perfetto_link=True):
            training_loop(mnist_data, params, n_layers, optimizer, qml_hybrid_loss_vag)
    else:
        loss, opt, update = training_loop_timed(mnist_data, params, n_layers, optimizer, opt_state, qml_hybrid_loss_vag)

    t1 = time.time()
    
    return t1 - t0, loss, opt, update