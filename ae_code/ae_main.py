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
                        vector (the code). Another list can be added for
                        transposed convolutional layers with the corresponding
                        out_channels. A 0 is added to include a fully connected
                        layer that is the same size as the first flattened
                        vector. 
                         The option exists to add fully connected layers at the 
                        very end.

    activations:        Defines an activation function for each layer. 
                         If only fully connected layers are used, each list 
                        must contain one more element than the 
                        corresponding ae_architecture list. If the 
                        corresponding list contains convolutional layers, 
                        the number of activation functions must equal
                        the number of all filters plus the fully connected
                        elements. 
    
    batch_size:         batch size

    kernel_size:        kernel size
    
    learning_rate:      learning rate
    
    epochs:             epochs
    
    print_reckonloss:   If True, numerical values for the reconstruction loss 
                        are printed

    save_results_in:    Ths specifies the relative path where results are saved. 


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
from helper_functions import num_elements
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch as pt
import math
from ae_and_train import build_model, prep_training, train_ae
from test_data import create_test_data
import copy
import itertools
import random


##------------------------- USER DEFINED PARAMETERS ------------------------##

save_results_in = "results/output/"     

code = 8
kernel_size = [100]

ae_architecture = [
#[[1, 6, 12], code, 0, [12, 6, 1], 1000],
[[1, 6, 12], code, 0, [12,6,1]],
[[1, 6, 12], code, 100],
[100,code,100],
]
activations = [
#[pt.relu, pt.relu, pt.tanh, pt.relu, pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],
[pt.relu, pt.relu, pt.tanh, pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],
[pt.relu, pt.relu, pt.relu, pt.relu, pt.tanh],
[pt.relu, pt.relu, pt.relu, pt.tanh],
]
batch_size = [10
]
learning_rate = [1e-4]

# global training settings
epochs = 100

# Set figure options for plotting
plot_every = 10
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['figure.figsize'] = (8, 4)
mpl.rcParams['axes.labelsize'] = "large"

# prints the reconstruction_loss if True
print_reconLoss = True

# define test function for input
def test_signal(x: pt.Tensor, t: float, z: complex) -> pt.Tensor:
    return pt.exp(-pt.pow((x-t/math.pi+0.5)/0.2, 2))
    #return (1 - pt.pow(x, 2)) * pt.exp(z*t)
    #return 2.0 * pt.tanh(5.0*x) / pt.cosh(5.0*x) * pt.exp(z*t)

input_signal = [[test_signal], [1.0j], [1.0]]

# Options related to reproducibility '''
seed = 42
pt.manual_seed(seed)
pt.backends.cudnn.benchmark = False
pt.backends.cudnn.deterministic = True


##---------------------------------- END -----------------------------------##


# creates folder to store files
if not os.path.exists(save_results_in):
    os.makedirs(save_results_in)

# Get matrix X that contains snapshots of the input signal at different
# time steps 
x, t, X = create_test_data(*input_signal)

# get size of a snapshot taken at any time t
[rows, columns] = X.shape
datapnts_at_t = rows

aemodels = [] 
loss_lists = []
previous_paras = [0,0,0,0,0]
same_except_kernel = 0
# Build autoencoder network and train it for each set of parameters
count_models = 0
for paras in itertools.product(ae_architecture, activations, kernel_size, 
                                                    learning_rate, 
                                                    batch_size):
    
    # Preselection: Number of activation functions must match corresponding 
    # number of layers
    if (num_elements(paras[0]) == len(paras[1]) and \
                             type(paras[0][0]) == list) \
        or \
            (len(paras[0]) == len(paras[1])-1 and type(paras[0][0]) == int):

        if type(paras[0][0]) == list:
            model, device = build_model(X, *paras[0:3]) 
            count_models += 1
            optimizer, criterion, train_loader = prep_training(X, model,
                                                            *paras[3:])
        
        # avoid multiple instances of the same fully connected model if 
        # only the kernel size is varied
        elif previous_paras[0:2] == paras[0:2] and \
                    previous_paras[3:] == paras[3:] and \
                    same_except_kernel == 1:
             break        
        
        else:
            model, device = build_model(X, *paras[0:3])
            optimizer, criterion, train_loader = prep_training(X, model,
                                                            *paras[3:])
            previous_paras = paras
            # same_except_kernel count in case the first two
            # fully connected layers are the same and previous_paras == 
            # [0,0,0,0,0]
            same_except_kernel += 1
            count_models += 1
    
        # train autoencoder with data from X
        train_dataset = X.real.T
        loss_values = train_ae(model, epochs, train_loader, datapnts_at_t, 
                                    optimizer, criterion, device,          
                                                print_reconLoss)

        aemodels.append(copy.deepcopy(model)) 
        loss_lists.append((loss_values))

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

# Load test data. Currently set to train data
test_dataset = train_dataset
test_loader = pt.utils.data.DataLoader(
test_dataset, batch_size=datapnts_at_t, shuffle=False
)

marker = itertools.cycle((',', '+', '.', 'o', '*'))
fig_loss = plt.figure()
ax = plt.subplot(111)

parameters_str = []
index = 0
same_except_kernel = 0
randomnumbers = []
for paras in itertools.product(ae_architecture, activations, kernel_size,
                                                learning_rate, batch_size):
    
    # Plot reconstruction loss only if number of activation functions matches
    # number of layers
    if (num_elements(paras[0]) == len(paras[1]) and type(paras[0][0]) == list) \
            or (len(paras[0]) == len(paras[1])-1 and type(paras[0][0]) == int):
        
        index += 1 

        model_activations_str = ''
        for model_activation_func in range(len(paras[1])): 
            activation_as_str = str(paras[1][model_activation_func].__name__)
                
            if model_activation_func < len(paras[1])-1:
                activation_as_str += ', '

            model_activations_str += activation_as_str
        
        parameters = []
        if type(paras[0][0]) == list:

            randomnumber = random.randint(1, 10000)
            randomnumbers.append(randomnumber)
            parameters = str(paras[0]) + ',\n[' + str(model_activations_str) + '],'\
                                    + '\nbatch_size= ' + str(paras[4]) \
                                    + ', epochs= ' + str(epochs) \
                                    + ', kernel_size= ' + str(paras[2]) \
                                    + ', learning_rate= ' + str(paras[3])
        
        elif previous_paras[0:2] == paras[0:2] and \
                    previous_paras[3:] == paras[3:] and \
                    same_except_kernel == 1:
             break        
        
        else:
            
            randomnumber = random.randint(1, 10000)
            randomnumbers.append(randomnumber)
            parameters = str(paras[0]) + ',\n[' + str(model_activations_str) + '],' \
                                    + '\nbatch_size= ' + str(paras[4]) \
                                    + ', epochs= ' + str(epochs) \
                                    + ', learning_rate= ' + str(paras[3])

            previous_paras = paras
            same_except_kernel += 1
            
        # store each models parameter string in a list for later use
        parameters_str.append(parameters) 
        ax.plot(loss_lists[index-1], label=('Model: ' + str(randomnumber) \
                                                            + '-' + str(index)),
                                                            marker=next(marker), ms=5, 
                                                            markevery=100)
        ax.legend()
        

        # save loss figure for each model
        with open(save_results_in + 'models.txt', 'a+'):
            

            file = open(save_results_in + 'models.txt',"r")
  
            readfile = file.read()

            # do not plot model twice
            if parameters not in readfile: 
                
                plt.figure()
                plt.plot(loss_lists[index-1], label=('Model: ' + str(randomnumber) \
                                                            + '-' + str(index)))    
                subtitle = 'Training: ' + str(randomnumber) + '-' + str(index)
                plt.title(subtitle)
                plt.savefig(save_results_in + 'loss: ' + str(randomnumber) 
                                            + '_' + str(index) + '.png')

ax.title.set_text('Training')

ax.set_xlabel('epochs')
ax.set_ylabel('loss')

plt.show()

X_recon = []
test_examples = None

with pt.no_grad():
    for aemodel in aemodels:
        for batch_features in test_loader:
            #batch_features = batch_features[0]
            test_examples = batch_features.view(-1, datapnts_at_t)
            X_recon_tensor = aemodel(test_examples)
            X_recon_tensor = X_recon_tensor.T
            X_recon.append(X_recon_tensor)
            #break





[rows, columns] = X_recon[0].shape
with pt.no_grad():
    # first plot original function 
    for i in range(0, columns, plot_every):
        plt.plot(x, X[:, i].real, color="k", alpha=1.0-0.01*i)
        plt.title('Original time series data')
        plt.savefig(save_results_in + 'original.png')
    # Plot each models output
    for index, X_rec in enumerate(X_recon):
        plt.show()
        for i in range(0, columns, plot_every):
            plt.plot(x, X_rec[:, i], color="k", alpha=1.0-0.01*i)

        subtitle = 'Model: ' + str(randomnumbers[index]) + '-' + str(index+1)
        plt.title(subtitle)

        with open(save_results_in + 'models.txt', 'a+'):
            

            file = open(save_results_in + 'models.txt',"r")
  
            readfile = file.read()

            # do not plot model twice
            if parameters_str[index] not in readfile: 
                    
                subtitle = 'Model: ' + str(randomnumbers[index]) + '-' + str(index+1)
                plt.title(subtitle)
                plt.savefig(save_results_in + str(randomnumbers[index]) 
                                            + '_' + str(index+1) + '.png')

            
file.close()

plt.show()

for i, aemodel in enumerate(aemodels):
    print('\n\nModel ' + str(randomnumbers[i]) + '-' + str(i+1) + ':\n\n', 
            'User input:\n\n', 
            parameters_str[i], '\n\n\n', aemodel,'\n') 

#for name, param in aemodel.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


with open(save_results_in + 'models.txt', 'a+'):
    for i, aemodel in enumerate(aemodels):

        file = open(save_results_in + 'models.txt',"r")
  
        readfile = file.read()

        # do not print model twice
        if parameters_str[i] not in readfile: 
            
            print('\n\nModel ' + str(randomnumbers[i]) + '-' + str(i+1) 
            + ':\n\n', parameters_str[i], '\n\n\n', aemodel,'\n',
            file = open(save_results_in + 'models.txt', 'a'))

file.close()


