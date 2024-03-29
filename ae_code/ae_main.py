'''
Build and evaluate autoencoder models.

All possible combinations of autoencoder models are build based on user input. 
Just add and change parameters in the user defined paramaters section.
 Training loss and output for each model are plotted for user defined 
input data. The results are saved in a user defined folder without 
overwriting previous results. A random id is given to each set of 
parameters to not mix them up. Results are not saved if they already exist.

Variables to be defined by the user:

    ae_architecture:  List defining the size of each fully connected
                        layer (ommiting in- and output layers). Each
                        integer element defines the number of neurons
                        per layer. 
                         If a list is added as the first element of a 
                        list in the ae_architecture list, the elements 
                        in this list define the number of filters in 
                        each convolutional layer. The final convolutional
                        layer is then automatically flattened and is followed
                        by fully connected layers including the latent 
                        vector (the code). To identify the code's position inside
                        the architecture, the autoencoder searches for the 
                        smallest number of neurons per layer.
                        Another list can be included for transposed convolutional 
                        layers with the corresponding out_channels. The number 0 
                        indicatese a fully connected layer that is the same size 
                        as the first flattened vector. 
                         Fully connected layers can be added at the very end.

    activations:        Defines an activation function for each layer. 
                         If only fully connected layers are used, each list 
                        must contain one more element than the 
                        corresponding ae_architecture list. If the 
                        corresponding list contains convolutional layers, 
                        the number of activation functions must equal
                        the number of all filters plus the fully connected
                        elements. 

    batch_size:         batch size

    code:               Latent vector length

    kernel_size:        kernel size
    
    learning_rate:      learning rate
    
    epochs:             epochs

    time_mapping:       If True Y replaces X as the target output of the 
                        autoencoder.
    
    print_reckonloss:   If True, numerical values for the reconstruction loss 
                        are printed

    save_results_in:    Ths specifies the relative path where results are saved. 

    X:                  Input data matrix containing snapshots

    Y:                  Output target matrix containing snapshots corresponding
                        to X = X(t+dt).


Example:

    4 (batch_size 5) + 4 (batch_size 10) = 8 autoencoder models will be build
    with the following parameters.
    (If, let's say, the order of the ae_architectures of lists is swapped it 
    does not matter. 
    If only one of the activation functions is given then only two models will 
    be build):
    
    code = 10
    kernel_size = [100]
    
    ae_architecture = [
    [[1, 6, 12], code, 0, [12, 6, 1], 1000],        # a
    [[1, 6, 12], code, 0, [12,6,1]],                # b
    [[1, 6, 12], code, 100],                        # c
    [80,code,80],                                   # d
    ]
    
    activations = [
    [pt.relu, pt.relu, pt.tanh, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh], # a
    [pt.relu, pt.relu, pt.tanh, pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],          # b
    [pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],                                     # c
    [pt.relu, pt.relu, pt.relu, pt.tanh],                                              # d
    ]
    
    batch_size = [5, 10]
    learning_rate = [1e-4]

'''

import os
import time
from flowtorch import data
from helper_functions import num_elements
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch as pt 
import math
from ae_and_train import build_model, prep_training, train_ae, \
                                                    count_parameters
import load_data
import copy
import itertools
import random
from visualization import plot_data_matrix, animate_flow, plot_code, plot_loss
import pickle
#import torchvision # MNIST testing


##------------------------ Input data -------------------------------------##

data_type = "1d_function"  # "openFoam"
# (field, component)
field = ("U",0)


# path to simulation data
path = "/mnt/d/Studium/Studienarbeit/Daten/datasets/of_cylinder2D_binary"             
#path = "/mnt/d/Studium/Studienarbeit/Daten/cylinder2D_base_100_100"
#path = "../cylinder2D_base_100_100/"

# define test function for input
def test_signal(x: pt.Tensor, t: float, z: complex) -> pt.Tensor:
    return pt.exp(-pt.pow((x-t/math.pi+0.5)/0.2, 2))
    #return (1 - pt.pow(x, 2)) * pt.exp(z*t)
    #return 2.0 * pt.tanh(5.0*x) / pt.cosh(5.0*x) * pt.exp(z*t)

def signal_one(x: pt.Tensor, t: float, z: complex) -> pt.Tensor:
    return (1 - pt.pow(x, 2)) * pt.exp(z*t)  

def signal_four(x: pt.Tensor, t: float, z: complex) -> pt.Tensor:
    return pt.exp(-pt.pow((x-t/math.pi+0.5)/0.2, 2))

input_signal = [[test_signal], [1.0j], [1.0]]
#input_signal = [signal_one, signal_four], [1.0j, 1.0j], [1.0, 1.0]

##------------------------- USER DEFINED PARAMETERS ------------------------##

save_results_in = "results/1d/fully_connected/testruns_tanh/"     

# autoencoder finds a representation for snapshots > t_start seconds
t_start = 4.0

code = 2
kernel_size = [50,
]

ae_architecture = [
#[[1, 6, 12], code, 0, [12, 6, 1], 1000],
#[[1, 6], 10, code, 0, [6, 1]],
#[[1, 6, 12], 100, code, 100],
#[400, 200, 120, 80, 40, code, 40, 80, 120, 200, 400],
#[200, 12, code, 40, 200],
[100, 1, 100],
[100, 100, 1],
[200, 100, 1],
[200, 200, 1],
[200, 200, 1, 200, 200],
[100, 100, 1, 100],
[200, 200, 1, 200],
[400, 1, 400],
[50, 1, 50],
[100, 50, 1, 50, 100]
#[1600, 1, 1600],
#[2400, 1, 2400],
#[1],
#[2],
#[20],
#[4],
]

activations = [
[pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],
[pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh],
[pt.tanh, pt.tanh, pt.tanh, pt.tanh],
[pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh], 
[pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.tanh, pt.selu, pt.selu],
]

batch_size = [10, 100
]

learning_rate = [1e-2, 1e-3, 1e-4
]

multi_decoder = [False,
]

time_mapping = [False,
]


# global training settings
epochs = 60000


# Set figure options for plotting
plot_every_vertex = 2
plot_every_snapshot = 10

# if datatype == openfoam this sets the contour min, max values
vmin = -0.7
vmax = 0.7

mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['figure.figsize'] = (8, 4)
mpl.rcParams['axes.labelsize'] = "large"

# prints the reconstruction loss if True
print_reconLoss = True
# animate flow in video
animation = True
# print these reconstructed snapshots
snapshots = (0,50,-1)

# Options related to reproducibility '''
seed = 42
pt.manual_seed(seed)
pt.backends.cudnn.benchmark = False
pt.backends.cudnn.deterministic = True


##---------------------------------- END -----------------------------------##

# creates folder to store files
if not os.path.exists(save_results_in):
    os.makedirs(save_results_in)

seed = pt.initial_seed()

X = None
x = None
y = None

if data_type == "1d_function":
  # Get matrix X that contains snapshots of the input signal at different
  # time steps 
  x, t, X = load_data.create_1d_funct_data(*input_signal)
  field = ("1d_function")
elif data_type == "openFoam":
  x, y, X = load_data.load_openfoam_data(path, field, t_start, plot_every_vertex)


X = X[:, 0:-1]
[rows, columns] = X.shape
datapnts_at_t = rows

# MNIST testing
#datapnts_at_t = 784

aemodels = [] 
code = []
loss_lists = []
epoch_list = []
stop_time_list = []
time_per_epoch_list = []
previous_paras = [0,0,0,0,0]
same_except_kernel = 0
# Build autoencoder network and train it for each set of parameters
count_models = 0
for paras in itertools.product(ae_architecture, activations, kernel_size, 
                                                        multi_decoder,
                                                        learning_rate, 
                                                        batch_size,
                                                        time_mapping):
    
    # Preselection: Number of activation functions must match corresponding 
    # number of layers
    if (num_elements(paras[0]) == len(paras[1]) and \
                             type(paras[0][0]) == list) \
        or \
            (len(paras[0]) == len(paras[1])-1 and type(paras[0][0]) == int):

        Y = load_data.set_mapping_target(X, paras[6])

        if type(paras[0][0]) == list:
            print(count_models+1)
            model, device = build_model(X, *paras[0:4]) 
            count_models += 1
            optimizer, criterion, train_loader = prep_training(X, model,
                                                            *paras[4:6], 
                                                            data_type, Y)
        
        # avoid multiple instances of the same fully connected model if 
        # only the kernel size is varied
        elif previous_paras[0:2] == paras[0:2] and \
                    previous_paras[3:] == paras[3:] and \
                    same_except_kernel == 1:
             continue        
        
        else:
            print(f"Model: {count_models+1}")
            model, device = build_model(X, *paras[0:4])
            optimizer, criterion, train_loader = prep_training(X, model,
                                                            *paras[4:6], 
                                                            data_type,
                                                            Y)
            previous_paras = paras
            # same_except_kernel count in case the first two
            # fully connected layers are the same and previous_paras == 
            # [0,0,0,0,0]
            same_except_kernel += 1
            count_models += 1

        tic= time.perf_counter()
        # train autoencoder with data from X
        loss_values, code, dec_out_list, epoch = train_ae(model, epochs, 
                                                                train_loader, 
                                                    datapnts_at_t, optimizer, 
                                                    criterion, device, 
                                                    print_reconLoss, paras[6])

        # measure training time
        toc = time.perf_counter()
        duration = toc - tic
        time_per_epoch = duration / epoch
        stop_time_list.append(duration)
        time_per_epoch_list.append(time_per_epoch)
        
        aemodels.append(copy.deepcopy(model)) 
        loss_lists.append((loss_values))
        epoch_list.append(epoch)

# Let user know why no model is build if number of activation functions
# do not correspond to the correct number of layers. In that case throw an error!
if count_models == 0:
    for paras in itertools.product(ae_architecture, activations, kernel_size,
                                                learning_rate, batch_size):
    
        if (num_elements(paras[0]) == len(paras[1])-1 and \
                                        type(paras[0][0]) == list) \
            or \
                (len(paras[0]) == len(paras[1])-2 and type(paras[0][0]) == int):
            
            raise ValueError("No activation function for final \
                            (reconstruction) layer given!")
        else:
            raise ValueError("Number of activations must equal number of \
                            hidden layers + output layer!")


marker = itertools.cycle((',', '+', '.', 'o', '*'))
#fig = plt.figure()
#ax = fig.add_subplot(111)

parameters_str = []
index = 0
same_except_kernel = 0
randomnumbers = []
for paras in itertools.product(ae_architecture, activations, kernel_size,
                                                        multi_decoder, 
                                                        learning_rate, 
                                                        batch_size,
                                                        time_mapping):
    
    # Plot reconstruction loss only if number of activation functions matches
    # number of layers
    if (num_elements(paras[0]) == len(paras[1]) and type(paras[0][0]) == list) \
            or (len(paras[0]) == len(paras[1])-1 and type(paras[0][0]) == int):
        
        index += 1 

        model_activations_str = ''
        for model_activation_func in range(len(paras[1])): 
            
            activation_as_str = ''
            if paras[1][model_activation_func] == 'lin':
                activation_as_str = 'lin'
            else:
                activation_as_str = str(paras[1][model_activation_func].__name__)
                
            if model_activation_func < len(paras[1])-1:
                activation_as_str += ', '

            model_activations_str += activation_as_str
        
        parameters = 'none'
        if type(paras[0][0]) == list:

            randomnumber = random.randint(1, 10000)
            randomnumbers.append(randomnumber)
            parameters = str(paras[0]) + ',\n[' + str(model_activations_str) + '],'\
                                    + '\nbatch_size = ' + str(paras[5]) \
                                    + ', epochs = ' + str(epochs) \
                                    + ', kernel_size = ' + str(paras[2]) \
                                    + ', learning_rate = ' + str(paras[4]) \
                                    + ', multi_decoder = ' + str(paras[3]) \
                                    + ',\n' \
                                    + 'time_mapping = ' + str(paras[6]) \
                    + ', data_type = ' + data_type + ', Input data= ' + str(field)
        
        elif previous_paras[0:2] == paras[0:2] and \
                    previous_paras[3:] == paras[3:] and \
                    same_except_kernel == 1:
             break        
        
        else:
            
            randomnumber = random.randint(1, 10000)
            randomnumbers.append(randomnumber)
            parameters = str(paras[0]) + ',\n[' + str(model_activations_str) + '],' \
                                    + '\nbatch_size = ' + str(paras[5]) \
                                    + ', epochs = ' + str(epochs) \
                                    + ', learning_rate = ' + str(paras[4]) \
                                    + ', multi_decoder = ' + str(paras[3]) + ',\n' \
                                    + 'time_mapping = ' + str(paras[6]) \
                        + ', data_type = ' + data_type + ', Input data= ' + str(field)

            previous_paras = paras
            same_except_kernel += 1
            
        # store each models parameter string in a list for later use
        parameters_str.append(parameters) 

        label = 'Model: ' + str(randomnumber) + '-' + str(index)
        title = r'$Training$'

        # save loss figure for each model
        with open(save_results_in + 'models.txt', 'a+'):
            

            file = open(save_results_in + 'models.txt',"r")
  
            readfile = file.read()

            # do not plot model twice
            if parameters not in readfile:    
                
                label = 'Model: ' + str(randomnumber) + '-' + str(index)
                subtitle = 'Training: ' + str(randomnumber) + '-' + str(index)

                fig2, ax2 = plot_loss(loss_lists[index-1], label, subtitle)

                ax2.get_figure().savefig(save_results_in + 'loss_' + str(randomnumber) 
                                            + '_' + str(index) + '.pdf')


# Load test data. Currently set to train data
train_dataset = None
if data_type == "1d_function":
    train_dataset = X.real.T
elif data_type == "openFoam":
    train_dataset = X.T
    
test_dataset = train_dataset
test_loader = pt.utils.data.DataLoader(
test_dataset, batch_size=datapnts_at_t, shuffle=False
)

## MNIST testing ------------
#transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#test_dataset = torchvision.datasets.MNIST(
#    root="~/torch_datasets", train=False, transform=transform, download=True
#)
## ----------------

test_loader = pt.utils.data.DataLoader(
    test_dataset, batch_size=datapnts_at_t, shuffle=False
)

X_recon = []
codes = []
dec_out_list = []
test_examples = None

with pt.no_grad():
    for aemodel in aemodels:
        for batch_features in test_loader:
            #batch_features = batch_features[0] # MNIST test
            test_examples = batch_features.view(-1, datapnts_at_t)
            X_recon_tensor, code_tensor, dec_outs = aemodel(test_examples)
            X_recon_tensor = X_recon_tensor.T  # MNIST test
            code_tensor = code_tensor.T
            
            dec_out_tensors = []
            for dec_out_tensor in dec_outs:
                dec_out_tensor = dec_out_tensor.T
                dec_out_tensors.append(dec_out_tensor)

            dec_out_list.append(dec_out_tensors)
            X_recon.append(X_recon_tensor)
            codes.append(code_tensor)
            break

## MNIST testing--------------------
#with pt.no_grad():
#    for aemodel in aemodels:

#        number = 10
#        plt.figure(figsize=(20, 4))

 #       for index in range(number):
 #           # display original
 #           ax = plt.subplot(2, number, index + 1)
 #           plt.imshow(test_examples[index].numpy().reshape(28, 28))
 #           plt.gray()
 #           ax.get_xaxis().set_visible(False)
 #           ax.get_yaxis().set_visible(False)

            # display reconstruction
#            ax = plt.subplot(2, number, index + 1 + number)
#            plt.imshow(X_recon[0][index].numpy().reshape(28, 28))
#            plt.gray()
#            ax.get_xaxis().set_visible(False)
#            ax.get_yaxis().set_visible(False)
#        plt.show()

# -----------------------------------

colorbar_label = ''
if data_type == "openFoam":
    colorbar_label = '$' + field[0] + '_' + str(field[1]+1) + '$'
    if field[0] == "U" and field[1] == 0:
        colorbar_label = '$u$'
    elif field[0] == "U" and field[1] == 1:
        colorbar_label = '$v$'

[rows, columns] = X_recon[0].shape

# Set up formatting for movie files
Writer = mpl.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='nformella'), bitrate=7200)

with pt.no_grad():
    # first plot original data
    if data_type == "1d_function":

        subtitle = "$Original time series data$"

        fig, ax = plot_data_matrix(X, x, y, data_type, subtitle,
                                                plot_every_snapshot)
            
        ax.get_figure().savefig(save_results_in + 'original.pdf')

    elif data_type == "openFoam":
        
        for idx, snapshot in enumerate(snapshots):
                                
            if snapshot == -1:
                snapshot = columns - 1
            
            title = "Original data (snapshot " + str(snapshot) + ')'
            fig, ax = plot_data_matrix(X, x, y, data_type, title,
                                                plot_every_snapshot, 
                                                snapshot,
                                                plot_every_vertex,
                                                colorbar_label,
                                                vmin, vmax)
        
            ax.get_figure().savefig(save_results_in + 'Original_snapshot_' 
                                                + str(snapshot)
                                                + '.pdf')

        if animation == True:

            subtitle = "Original flow"
            frames=range(0, columns, plot_every_snapshot)
            anim_orig = animate_flow(X, x, y, frames, subtitle, plot_every_vertex,
                                                                    vmin, vmax)
            
            anim_orig.save(save_results_in + 'orig.mp4', writer=writer)
                                                                            
    
    anim_recon = None
    # Plot each model's output
    for index, X_rec in enumerate(X_recon):

        if data_type == "1d_function":

            subtitle = '$Model: ' + str(randomnumbers[index]) + '-' \
                                                        + str(index+1) + '$'    

            fig, ax = plot_data_matrix(X_rec, x, y, data_type, subtitle,
                                                plot_every_snapshot)

            with open(save_results_in + 'models.txt', 'a+'):
            

                file = open(save_results_in + 'models.txt',"r")
    
                readfile = file.read()

                # do not plot model twice
                if parameters_str[index] not in readfile: 

                    ax.get_figure().savefig(save_results_in 
                                                + str(randomnumbers[index]) 
                                                    + '_' + str(index+1) 
                                                    + '_recon.pdf')  
                    pt.save(X_rec, save_results_in 
                                    + str(randomnumbers[index]) 
                                    + '_' + str(index+1) 
                                    + '_recon.pt')

            file.close()

        elif data_type == "openFoam":

            ax_arr = []
            for idx, snapshot in enumerate(snapshots):
                                            
                if snapshot == -1:
                    snapshot = columns - 1

                subtitle = 'Model: ' + str(randomnumbers[index]) \
                                                    + '-' + str(index+1) \
                                                    + ' (snapshot '  \
                                                    + str(snapshot) + ')'

                fig, ax = plot_data_matrix(X_rec, x, y, data_type, subtitle,
                                                plot_every_snapshot, snapshot,
                                                plot_every_vertex, 
                                                colorbar_label, vmin, vmax)
                    
                ax_arr.append(ax)  


            if animation == True:
                
                subtitle = 'Model: ' + str(randomnumbers[index]) + '-' \
                                                            + str(index+1) \
                                                        
                frames=range(0, columns, plot_every_snapshot)
                anim_recon = animate_flow(X_rec, x, y, frames, subtitle, 
                                                    plot_every_vertex,
                                                    vmin, vmax)

        
            with open(save_results_in + 'models.txt', 'a+'):

                file = open(save_results_in + 'models.txt',"r")
    
                readfile = file.read()

                # do not plot model twice
                if parameters_str[index] not in readfile: 

                    pt.save(X_rec, save_results_in + str(randomnumbers[index]) 
                                                    + '_' + str(index+1) 
                                                    + '_recon.pt')

                    if animation == True:
                        
                        anim_recon.save(save_results_in + str(randomnumbers[index]) 
                                             + '_' + str(index+1) + '_recon.mp4', 
                                                                    writer=writer)
                    
                    for idx, snapshot in enumerate(snapshots):
                        if snapshot < columns:

                            if snapshot == -1:
                                snapshot = columns - 1

                            subtitle_snap = 'Model: ' + str(randomnumbers[index]) \
                                                        + '-' + str(index+1) \
                                                        + ' (snapshot '  \
                                                        + str(snapshot) + ')'

                            ax_arr[idx].set_title(subtitle_snap)

                            ax_arr[idx].get_figure().savefig(save_results_in
                                                    + str(randomnumbers[index]) 
                                                    + '_' + str(index+1) 
                                                    + '_snapshot_' + str(snapshot)
                                                    + '.pdf') 
                             
            file.close()


# Latent feature visualization
with pt.no_grad():
    # Plot each model's latent features
    axes = []
    for index, code in enumerate(codes):

        
        title = "Latent features: Model "  + \
                str(randomnumbers[index]) + '-' + str(index+1)

        fig, ax = plot_code(code, title)
        
        axes.append(ax)

        with open(save_results_in + 'models.txt', 'a+'):

            file = open(save_results_in + 'models.txt',"r")
  
            readfile = file.read()

            # do not plot model twice
            if parameters_str[index] not in readfile: 
                    
                axes[index].set_title(title)

                axes[index].get_figure().savefig(save_results_in + 'code_' 
                                                + str(randomnumbers[index]) 
                                                + '_' + str(index+1) + '.pdf')  
                pt.save(code, save_results_in + 'code_' 
                                                + str(randomnumbers[index]) 
                                                + '_' + str(index+1) + '.pt')
        
        file.close()

plt.close('all')


# Multiple decoder output
with pt.no_grad():
    
    anim_recon = None
    # Plot each model's output
    for index, dec_outs in enumerate(dec_out_list):

        if data_type == "1d_function":
            axes = []
            for i, dec_out in enumerate(dec_outs):

                if len(dec_outs) == 1:
                    break

                subtitle = 'Model: ' + str(randomnumbers[index]) + '-' \
                                        + str(index+1) + ' (Decoder: ' \
                                        + str(i+1)

                fig, ax = plot_data_matrix(dec_out, x, y, data_type, 
                                        subtitle, plot_every_snapshot,
                                        colorbar_label, vmin, vmax)
                
                axes.append(ax)

                axes[i].get_figure().savefig(save_results_in 
                                                + str(randomnumbers[index]) 
                                                + '_' + str(index+1) 
                                                + '_decoder(' + str(i+1) + ').pdf')  

            with open(save_results_in + 'models.txt', 'a+'):
            

                file = open(save_results_in + 'models.txt',"r")
    
                readfile = file.read()

                # do not plot model twice
                if parameters_str[index] not in readfile: 
                    
                    for i, dec_out in enumerate(dec_outs):

                        if len(dec_outs) == 1:
                            break
                        
                        axes[i].get_figure().savefig(save_results_in 
                                                + str(randomnumbers[index]) 
                                                + '_' + str(index+1) 
                                                + '_decoder(' + str(i+1) + ').pdf')  


            file.close()

        elif data_type == "openFoam":
            
            ax_arr = []
            for i, dec_out in enumerate(dec_outs):
                
                if len(dec_outs) == 1:
                    break
                
                for idx, snapshot in enumerate(snapshots):

                    if snapshot == -1:
                        snapshot = columns - 1


                    subtitle = 'Model: ' + str(randomnumbers[index]) + '-' \
                                        + str(index+1) + ' (Decoder: ' \
                                        + str(i+1) + ', snapshot: ' + str(snapshot)

                    fig, ax = plot_data_matrix(dec_out, x, y, data_type, subtitle,
                                                    plot_every_snapshot, snapshot,
                                                    plot_every_vertex, 
                                                    colorbar_label, vmin, vmax)

                    ax_arr.append(ax)  


                if animation == True:

                    subtitle = 'Model: ' + str(randomnumbers[index]) + '-' \
                                                + str(index+1) + ' (Decoder: ' \
                                                + str(i+1)
                    frames=range(0, columns, plot_every_snapshot)
                    anim_recon = animate_flow(dec_out, x, y, frames, subtitle, 
                                                    plot_every_vertex, 
                                                    vmin, vmax)


                with open(save_results_in + 'models.txt', 'a+'):

                    file = open(save_results_in + 'models.txt',"r")

                    readfile = file.read()

                    # do not plot model twice
                    if parameters_str[index] not in readfile: 

                        if animation == True:

                            anim_recon.save(save_results_in + str(randomnumbers[index]) 
                                                 + '_' + str(index+1) + '(Decoder(' 
                                                 + str(i+1) + ')).mp4', 
                                                writer=writer)

                        for idx, snapshot in enumerate(snapshots):
                            if snapshot < columns:

                                if snapshot == -1:
                                    snapshot = columns - 1

                                subtitle_snap = 'Model: ' + str(randomnumbers[index]) \
                                                            + '-' + str(index+1) \
                                                            + ' (Decoder(' + str(i+1) + '), ' \
                                                            + 'snapshot '  \
                                                            + str(snapshot)

                                ax_arr[idx].set_title(subtitle_snap)

                                ax_arr[idx].get_figure().savefig(save_results_in
                                                        + str(randomnumbers[index]) 
                                                        + '_' + str(index+1) 
                                                        + '_(snapshot(' + str(snapshot) 
                                                        + ')Decoder(' + str(i+1)
                                                        + ')).pdf') 

                file.close()

plt.close('all')

# write to text file
with open(save_results_in + 'models.txt', 'a+'):
    for i, aemodel in enumerate(aemodels):

        file = open(save_results_in + 'models.txt',"r")
        readfile = file.read()

        # do not print model twice
        if parameters_str[i] not in readfile: 
            
            total_params, table = count_parameters(aemodel)

            show_loss = []
            if len(loss_lists[i]) > 3:
                show_loss.append(loss_lists[i][-3:])
            else:
                show_loss.append(loss_lists[i][-len(loss_lists[i]):])

            with open(save_results_in + f"loss_{randomnumbers[i]}_{i+1}.txt", "wb") as fp:
                pickle.dump(loss_lists[i], fp)

            pt.save(aemodel.state_dict(), save_results_in + str(randomnumbers[i]) + '.pt')

            print('\n\nModel ' + str(randomnumbers[i]) + '-' + str(i+1) + ':\n\n\n', 
                    parameters_str[i], '\n\n', f"Trained {epoch_list[i]} epochs\n",
                    'Total training time spent: ', str(round(stop_time_list[i],4)), ' s\n',
                    'Time per epoch: ', str(round(time_per_epoch_list[i],8)), ' s\n',
                    'Initial seed: ', str(seed), '\n',
                    'Epoch training loss (last values): ', '\n',
                    show_loss, '\n\n',
                    aemodel,'\n\n',
                    table, '\n', f"Total Trainable Params: {total_params}", '\n\n', 
                    '\n---------------------------------------------------------------------',
                    '\n---------------------------------------------------------------------',
                    sep='', 
                    file = open(save_results_in + 'models.txt', 'a'))

                                                                               
file.close()


for i, aemodel in enumerate(aemodels):
    print('\n\nModel ' + str(randomnumbers[i]) + '-' + str(i+1) + ':\n\n', 
            'User input:\n\n', 
            parameters_str[i], '\n\n', f"Trained {epoch_list[i]} epochs\n",
            'Total training time spent: ', str(round(stop_time_list[i],4)), ' s\n',
            'Time per epoch: ', str(round(time_per_epoch_list[i],8)), ' s\n',
            'Initial seed: ', str(seed), '\n\n',
            aemodel,'\n', sep='') 
    total_params, table = count_parameters(aemodel)
    print(table)
    print(f"Total Trainable Params: {total_params}\n\n")
    print('\n---------------------------------------------------------------------',
            '\n---------------------------------------------------------------------')
#for name, param in aemodel.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print('\nDone :-) !!')