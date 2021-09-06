import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mpl.use("Agg")


def animate(i, data_matrix, x, y, subtitle, plot_every_vertex):

    plt.cla()
    ax = plt.gca()
    ax.tricontourf(x[::plot_every_vertex], 
                                    y[::plot_every_vertex], 
                                    data_matrix[::plot_every_vertex, i])
    circ = plt.Circle((0.2, 0.2), 0.05, color='dimgrey')
    ax.add_patch(circ)
    ax.set_aspect("equal", 'box')
    plt.title(subtitle)



def animate_flow(data_matrix, x, y, frames, subtitle='Data', 
                                                plot_every_vertex=1):
    
    frames=frames
    fargs = (data_matrix, x, y, subtitle, plot_every_vertex)
    anim = FuncAnimation(plt.gcf(), animate, frames=frames, 
                                                interval=200000,
                                                fargs=fargs)
    
    plt.draw()
    plt.show()
    
    return anim



def plot_data_matrix(data_matrix, x, y, data_type, 
                                        subtitle="Data",
                                        plot_every_snapshot=1,
                                        snapshot=0,
                                        plot_every_vertex=1):

    [rows, columns] = data_matrix.shape

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if data_type == "1d_function":

        for i in range(0, columns, plot_every_snapshot):
            ax.plot(x, data_matrix[:, i], color="b", alpha=1.0-0.01*i)
        
        ax.set_title(subtitle)
        ax.set_xlabel('x')

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

        ax.tricontourf(x[::plot_every_vertex], 
                            y[::plot_every_vertex], 
                            data_matrix[::plot_every_vertex, snapshot])   

        circle = plt.Circle((0.2, 0.2), 0.05, color='dimgrey')
        ax.add_patch(circle)
        ax.set_aspect("equal", 'box')
        ax.set_title(subtitle)
        
        plt.show() 

    return fig, ax