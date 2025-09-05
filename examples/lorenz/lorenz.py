import jax.numpy as jnp

def lorenz_vector_field(t, state, params):
    """
    Vector field for the Lorenz system, suitable for use with diffrax.
    Args:
        t: time (unused, but required by diffrax signature)
        state: state dictionary
        params: dict with parameters (sigma, rho, beta)
    Returns:
        dydt: dictionary with time derivatives
    """
    sigma = params['sigma']
    rho = params['rho']
    beta = params['beta']
    x = state['x']
    y = state['y']
    z = state['z']

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return {'x': dxdt, 'y': dydt, 'z': dzdt}
