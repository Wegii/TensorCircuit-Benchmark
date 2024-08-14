import tensorcircuit as tc


def qml_ys(x, weights, nlayers):
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