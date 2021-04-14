'''
Modules and functions to build and train an autoencoder.

Classes:

    DenseAe(DenseAe, int, list, list)
    LinearBlock()

Functions:

    build_model(Tensor, list, list) -> DenseAe, device
    prep_training(Tensor, DenseAe, list, int) -> 
                                        optimizer, criterion, DataLoader
    train_ae(DenseAe, int, DataLoader, int, object, optimizer, criterion, 
                                                bool) -> list
    
'''

import torch.nn as nn
import torch as pt
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearBlock(nn.Module):
    """    
    Class adds a block with a linear layer followed by an activation function 
    to the neural network.

    ...

    Attributes
    ----------
    in_nums : int
        input features
    out_nums : int
        output features
    activation : function
        nonlinearity/activation function

    Methods
    -------
    forward(x):
        Passes input through linear block.
    """

    def __init__(self, in_nums, out_nums, activation):
        """
        Constructs necessary attributes for the LinearBlock with it's
        activation function.

        Parameters
        ----------
            in_nums : int
                input features
            out_nums : int
                output features
            activation : function
                nonlinearity/activation function        
        """

        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_nums, out_nums)
        self.activation = activation

    def forward(self, x):
        """
        Takes in x, passes it through the linear block and applies an
        activation function.
        """
        return self.activation(self.linear(x))


class DenseAe(nn.Module):
    def __init__(self, input_shape, neurons_per_layer, activations):
        super(DenseAe, self).__init__()

        neurons = neurons_per_layer.copy()
        neurons.append(input_shape)

         
        if len(neurons) == len(activations)-1:
            raise ValueError("No activation function for final \
                                    (reconstruction) layer given!")
        elif len(neurons)  != len(activations): 
            raise ValueError("Number of activations must equal number \
                                     of hidden layers + output layer!")
        
        # Construct a linear layer followed by an activation function
        self.blocks = nn.ModuleList()

        previous = input_shape
        for i in range(len(neurons)):
            self.blocks.append(LinearBlock(previous, neurons[i], 
                                                        activations[i]))
            previous = neurons[i]

    def forward(self, x):
        """
        Takes in x, returns it as the model output after being passed 
        through each block.
        """
        
        for block in self.blocks:
            x = block(x)

        return x


def build_model(X, neurons_per_layer, activations):

    #  use gpu if available
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # get size of one single snapshot
    [rows, columns] = X.shape
    datapnts_at_t = rows

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = DenseAe(datapnts_at_t, neurons_per_layer, activations).to(device)

    return model, device


def prep_training(X, model, learning_rate, batch_size=1):

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()

    # load train data
    train_dataset = X.real.T
    train_loader = pt.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )

    return optimizer, criterion, train_loader


''' function to train autoencoder for specified number of epochs '''
def train_ae(model, epochs, train_loader, input_shape, optimizer, 
                                criterion, device, print_reconLoss):

    loss_values = []
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # Reshape mini-batch data to [N, 784] matrix and load 
            # it to the active device
            batch_features = batch_features.view(-1, input_shape).to(device)
            
            # Reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes 
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        if print_reconLoss == True:
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

        loss_values.append(loss)
    
    return loss_values

