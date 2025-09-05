from jax import jit, vmap
import diffrax as dfx
import equinox as eqx
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike
from functools import partial

from .utils import load_yaml_of_distributions

def solve_ode(
    params: dict[ArrayLike],
    init_state: dict[ArrayLike],
    vector_field: callable, 
    times: ArrayLike, 
    solver=dfx.Tsit5(),
    dt0=0.01,
    stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
    **solver_kwargs
):
    """Light wrapper around diffrax's diffeqsolve function."""
    term = dfx.ODETerm(vector_field)
    saveat = dfx.SaveAt(ts=times)

    sol = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=times[0],
        t1=times[-1],
        dt0=dt0,
        y0=init_state,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        **solver_kwargs
    )
    return sol

class DynamicalSystem(eqx.Module):
    solver: callable
    param_dists: dict
    init_state_dists: dict
    _times: ArrayLike
    _vector_field: callable = eqx.static_field()
    _random_param_names: list[str] = eqx.static_field()
    _random_init_state_names: list[str] = eqx.static_field()

    def __init__(
        self,
        vector_field: callable,
        times: ArrayLike,
        param_dists: dict | str,
        init_state_dists: dict | str,
        solver_kwargs: dict = {},
    ):
        self.solver = jit(partial(
            solve_ode,
            vector_field=vector_field,
            times=times,
            **solver_kwargs
        ))
        if isinstance(param_dists, str):
            self.param_dists = load_yaml_of_distributions(param_dists)
        else:
            self.param_dists = param_dists
        if isinstance(init_state_dists, str):
            self.init_state_dists = load_yaml_of_distributions(init_state_dists)
        else:
            self.init_state_dists = init_state_dists
        self._times = times
        self._vector_field = vector_field
        self._random_param_names = [key for key in self.param_dists.keys() if isinstance(self.param_dists[key], dist.Distribution)]
        self._random_init_state_names = [key for key in self.init_state_dists.keys() if isinstance(self.init_state_dists[key], dist.Distribution)]

    def simulate(self, params: dict, init_state: dict) -> Array:
        sol = self.solver(params, init_state)
        return sol
    
    @property
    def times(self) -> ArrayLike:
        return self._times
    
    @property
    def vector_field(self) -> callable:
        return self._vector_field
    
    @property
    def random_param_names(self) -> list[str]:
        return self._random_param_names
    
    @property
    def random_init_state_names(self) -> list[str]:
        return self._random_init_state_names
    
    @property
    def random_input_names(self) -> list[str]:
        return self._random_param_names + self._random_init_state_names
    
    @property
    def all_param_names(self) -> list[str]:
        return list(self.param_dists.keys())

    @property
    def all_state_names(self) -> list[str]:
        return list(self.init_state_dists.keys())

    @property
    def all_names(self) -> list[str]:
        return list(self.param_dists.keys()) + list(self.init_state_dists.keys())
    
    @property
    def num_random_inputs(self) -> int:
        return len(self.random_input_names)
    
    def __repr__(self):
        # Create a simplified representation excluding the solver
        lines = []
        lines.append("DynamicalSystem(")
        lines.append(f"  param_dists={self.param_dists},")
        lines.append(f"  init_state_dists={self.init_state_dists},")
        lines.append(f"  _times=Array(shape={getattr(self._times, 'shape', 'unknown')}, dtype={getattr(self._times, 'dtype', 'unknown')}),")
        lines.append(f"  _vector_field={self._vector_field.__name__ if hasattr(self._vector_field, '__name__') else repr(self._vector_field)},")
        lines.append(f"  solver=<compiled_function>")
        lines.append(")")
        return "\n".join(lines)

class OutputSystem(eqx.Module):
    pointwise_output_fns: dict[str, callable] = eqx.field(default_factory=dict)
    functional_output_fns: dict[str, callable] = eqx.field(default_factory=dict)

    def __init__(
        self,
        pointwise_output_fns: dict[str, callable] = {},
        functional_output_fns: dict[str, callable] = {}
    ):
        self.pointwise_output_fns = pointwise_output_fns
        self.functional_output_fns = functional_output_fns

    def compute_pointwise_outputs(
        self, 
        times: Array, 
        states: Array,
        params: dict
    ) -> dict[str, Array]:
        outputs = {}
        for name, fn in self.pointwise_output_fns.items():
            outputs[name] = vmap(fn, (0, 0, None))(times, states, params)
        return outputs

    def compute_functional_outputs(
        self, 
        times: Array,
        states: Array, 
        params: dict
    ) -> dict[str, Array]:
        outputs = {}
        for name, fn in self.functional_output_fns.items():
            outputs[name] = fn(times, states, params)
        return outputs
    
    @property
    def pointwise_output_names(self) -> list[str]:
        return list(self.pointwise_output_fns.keys())

    @property
    def functional_output_names(self) -> list[str]:
        return list(self.functional_output_fns.keys())