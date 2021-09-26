from typing import Callable, List
import torch as pt
import matplotlib as mpl
import matplotlib.pyplot as plt
from flowtorch.data import FOAMDataloader, mask_box
import math


def create_1d_funct_data(
    signal_function: List[Callable[[pt.Tensor, float, complex], pt.Tensor]],
    time_behavior: List[complex],
    signal_weight: List[float],
    n_samples_time: int=100,
    n_samples_space: int=1000,
    sample_random: bool=False,
    noise_level: float=0.0
) -> pt.Tensor:

    assert len(signal_function) == len(time_behavior) and \
                    len(signal_function) == len(signal_weight),\
    "signal_function, time_behavior, und signal_weight must have \
    the same length."
    
    if sample_random:
        x = (pt.rand(n_samples_space) - 0.5) * 2.0
    else:
        x = pt.linspace(-1.0, 1.0, n_samples_space)
        
    t = pt.linspace(0.0, math.pi, n_samples_time)
        
    X = pt.zeros((n_samples_space, n_samples_time), dtype=pt.cfloat)
    
    for n, t_n in enumerate(t):
        for signal, z, weight in zip(signal_function, time_behavior, 
                                                        signal_weight):

            X[:, n] += weight * signal(x, t_n, z) + (pt.rand(n_samples_space) 
                                                    - 0.5) * 2.0 * noise_level
        
        X[:, n] += (pt.rand(n_samples_space) - 0.5) * 2.0 * noise_level
    
    return x, t, X


def load_openfoam_data(path, field, t_start=1, every=1):

    # increase resolution of plots
    mpl.rcParams['figure.dpi'] = 160

    loader = FOAMDataloader(path)
    fields = loader.field_names
    # select a subset of the available snapshots
    times = loader.write_times
    print(f"Number of available snapshots: {len(times)}")
    print("First five write times: ", times[:5])
    print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

    # load vertices and discard z-coordinate (2D case)
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])

    print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}\n")

    fig, ax = plt.subplots()
    ax.scatter(vertices[::every, 0], vertices[::every, 1], s=0.5, 
                                                    c=mask[::every])
    ax.set_aspect("equal", 'box')
    ax.set_xlim(0.0, 2.2)
    ax.set_ylim(0.0, 0.41)
    plt.show()

    window_times = [time for time in times if float(time) >= t_start]
    data_matrix = pt.zeros((mask.sum().item(), len(window_times)), 
                                                    dtype=pt.float32)

    for i, time in enumerate(window_times):
        # load the desired vector field, take the i-component [:, i], 
        # and apply the mask
        data_matrix[:, i] = pt.masked_select(loader.load_snapshot(
                                            field[0], time)[:,field[1]], mask)

    # subtract the temporal mean
    data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)

    x = pt.masked_select(vertices[:, 0], mask)
    y = pt.masked_select(vertices[:, 1], mask)

    # Plot scalar field with tricontour
    snapshot_nr = -1
    plt.tricontourf(x[::every], y[::every], data_matrix[::every, snapshot_nr])
    plt.title('Scalar Field (snapshot: ' + str(snapshot_nr) + ')')
    
    plt.show()
    plt.savefig('ScalarField.pdf')

    return x, y, data_matrix



def set_mapping_target(X, time_mapping):
    
    # get snapshot taken at any time t + dt by shifting input snapshots
    Y = X[:, 1:]
    if time_mapping == False:
        Y = None
    
    return Y