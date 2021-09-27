import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mpl.use("Agg")
from matplotlib.cm import ScalarMappable



def animate(i, data_matrix, x, y, subtitle, plot_every_vertex,
                                                        vmin=None,
                                                        vmax=None):

    plt.cla()
    ax = plt.gca()
    contour_plot = ax.tricontourf(x[::plot_every_vertex], 
                                    y[::plot_every_vertex], 
                                    data_matrix[::plot_every_vertex, i],
                                    levels=25, cmap='jet', 
                                    vmin=vmin, vmax=vmax)
    circ = plt.Circle((0.2, 0.2), 0.05, color='dimgrey')
    ax.add_patch(circ)
    ax.set_aspect("equal", 'box')
    
    plt.title(subtitle)



def animate_flow(data_matrix, x, y, frames, subtitle=r'$Data$', 
                                                plot_every_vertex=1,
                                                vmin=None, vmax=None):
    
    frames=frames
    fargs = (data_matrix, x, y, subtitle, plot_every_vertex, vmin, vmax)
    anim = FuncAnimation(plt.gcf(), animate, frames=frames, 
                                                interval=200000,
                                                fargs=fargs)
    
    plt.draw()
    plt.show()
    
    return anim



def plot_data_matrix(data_matrix, x, y, data_type, 
                                        subtitle=r"$Data$",
                                        plot_every_snapshot=1,
                                        snapshot=0,
                                        plot_every_vertex=1,
                                        label=None,
                                        vmin=None, vmax=None):

    [rows, columns] = data_matrix.shape

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if data_type == "1d_function":

        for i in range(0, columns, plot_every_snapshot):
            ax.plot(x, data_matrix[:, i], color="k", alpha=1.0-0.01*i)
        
        ax.set_title(subtitle)
        ax.set_xlabel(r'$x$')

        # set axis limits that are likely to capture the entire
        # range of values (also for future models)
        upper = 1.1
        lower = 0
        left = -1
        right = 1

        bottom_new, top_new = ax.get_ylim()
        left_new, right_new = ax.get_xlim()
        # override limits if necessary
        if top_new > upper:
            ax.set_ylim(top=top_new)
        elif bottom_new < lower:
            ax.set_ylim(bottom=bottom_new)
        else:
            ax.set_ylim(lower, upper)
        
        if left_new < left:
            ax.set_xlim(left=left_new)
        elif right_new > right:
            ax.set_xlim(right=right_new)
        else:
            ax.set_xlim(left, right)
        
        plt.show()

    elif data_type == "openFoam":
        
        if snapshot >= columns:
            
            raise ValueError("Snapshot " + str(snapshot) + " out of range")

        #levels = 25
        contour_plot = ax.tricontourf(x[::plot_every_vertex], 
                                y[::plot_every_vertex], 
                                data_matrix[::plot_every_vertex, snapshot],
                                levels=25, cmap='jet', vmin=vmin, vmax=vmax)   

        circle = plt.Circle((0.2, 0.2), 0.05, color='dimgrey')
        ax.add_patch(circle)
        ax.set_aspect("equal", 'box')
        ax.set_title(subtitle)
        
        ax.get_figure().colorbar(ScalarMappable(norm=contour_plot.norm, 
                                                cmap=contour_plot.cmap),
                                                label=label)
        plt.show() 

    return fig, ax


def plot_code(code, title):


    fig = plt.figure()
    ax = fig.add_subplot(111)

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    [rows, columns] = code.shape
    x = list(range(1, columns+1))

    for i in range(0, rows):
        ax.plot(x, code[i], color="k", label=r'$neuron $' 
                                    + str(i+1), marker=next(marker))

    ax.set_title(title)
    ax.set_xlabel(r'$t$')
    ax.legend()

    plt.show()

    return fig, ax
    

def plot_loss(loss, label, subtitle='$Training$', ax=None, marker=None):

    if ax == None:
        plt.cla()
        ax = plt.subplot(111)

    fig = ax.get_figure()
    ax.semilogy(loss, label=label, marker=marker, ms=5, markevery=100)
    
    ax.legend()

    ax.set_title(subtitle)
    ax.set_xlabel(r'$epochs$')
    ax.set_ylabel(r'$loss$')

    plt.show()
    
    return fig, ax

