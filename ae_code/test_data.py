from typing import Callable, List
import torch as pt
import math

def create_test_data(
    signal_function: List[Callable[[pt.Tensor, float, complex], pt.Tensor]],
    time_behavior: List[complex],
    signal_weight: List[float],
    n_samples_time: int=100,
    n_samples_space: int=1000,
    sample_random: bool=False,
    noise_level: float=0.0
) -> pt.Tensor:

    assert len(signal_function) == len(time_behavior) and len(signal_function) == len(signal_weight),\
    "signal_function, time_behavior, und signal_weight must have the same length."
    
    if sample_random:
        x = (pt.rand(n_samples_space) - 0.5) * 2.0
    else:
        x = pt.linspace(-1.0, 1.0, n_samples_space)
        
    t = pt.linspace(0.0, math.pi, n_samples_time)
        
    X = pt.zeros((n_samples_space, n_samples_time), dtype=pt.cfloat)
    
    for n, t_n in enumerate(t):
        for signal, z, weight in zip(signal_function, time_behavior, signal_weight):
            X[:, n] += weight * signal(x, t_n, z) + (pt.rand(n_samples_space) - 0.5) * 2.0 * noise_level
        X[:, n] += (pt.rand(n_samples_space) - 0.5) * 2.0 * noise_level
    return x, t, X