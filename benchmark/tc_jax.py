import tensorcircuit as tc
import jax
from jax.config import config
import optax
import time
from circuit import qml_ys
from utils import load_mnist


def qml_hybrid_loss(x, y, params, nlayers):
    weights = params["qweights"]
    w = params["cweights:w"]
    b = params["cweights:b"]
    ypred = qml_ys(x, weights, nlayers)
    ypred = tc.backend.reshape(ypred, [-1, 1])
    ypred = w @ ypred + b
    ypred = jax.nn.sigmoid(ypred)
    ypred = ypred[0]
    loss = -y * tc.backend.log(ypred) - (1 - y) * tc.backend.log(1 - ypred)
    return loss

def tc_jax_benchmark():
    nlayers = 3

    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, num=4)
    params = {
        "qweights": jax.random.normal(subkeys[0], shape=[nlayers * 2, 9]),
        "cweights:w": jax.random.normal(subkeys[1], shape=[9]),
        "cweights:b": jax.random.normal(subkeys[2], shape=[1]),
    }

    qml_hybrid_loss_vag = tc.backend.jit(
        tc.backend.vvag(qml_hybrid_loss, vectorized_argnums=(0, 1), argnums=2),
        static_argnums=3,
    )

    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(params)
    mnist_data = load_mnist()

    for i, (xs, ys) in zip(range(2000), mnist_data):  # using tf data loader here
        xs = xs.numpy()
        ys = ys.numpy()
        v, grads = qml_hybrid_loss_vag(xs, ys, params, nlayers)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 30 == 0:
            print(jnp.mean(v))