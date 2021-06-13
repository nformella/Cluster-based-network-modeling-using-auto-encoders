'''
Modules and functions to build and train an autoencoder.

Classes:

    AE(AE, int, list, list)
    LinearBlock()

Functions:

    build_model(Tensor, list, list) -> AE, device
    prep_training(Tensor, AE, list, int) -> 
                                        optimizer, criterion, DataLoader
    train_ae(AE, int, DataLoader, int, object, optimizer, criterion, 
                                                bool) -> list
    
'''

from helper_functions import num_elements
import torch.nn as nn
import torch as pt
import torch.optim as optim
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                                                        activation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_activation = activation

    def forward(self, x):
        return self.conv_activation(self.conv(x))


class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                                                        activation):
        super(ConvTBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        self.conv_activation = activation

    def forward(self, x):
        return self.conv_activation(self.conv(x))


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
        Constructs necessary attributes for the LinearBlock with its
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

        Parameters
        ----------
            in_nums : int
                input features
            out_nums : int
                output features
            activation : function
                nonlinearity/activation function
        """
        return self.activation(self.linear(x))


class AE(nn.Module):
    def __init__(self, input_shape, architecture, activations, 
                                                kernel_size=100):
        super(AE, self).__init__()

        self.layers = architecture.copy()
        self.layers.append(input_shape)

 
        if (num_elements(self.layers) == len(activations)-1 and \
                type(self.layers[0]) == list) \
            or \
                (len(self.layers) == len(activations)+1 and \
                type(self.layers[0]) == int):
            
            raise ValueError("No activation function for final \
                                    (reconstruction) layer given!")

        elif (num_elements(self.layers) != len(activations)+1 and \
                                        type(self.layers[0]) == list) \
            or \
                (len(self.layers) != len(activations) and \
                                        type(self.layers[0]) == int):

            raise ValueError("Number of activations must equal \
                        number of hidden self.layers + output layer!")
        
        # Construct each layer followed by an activation function
        self.blocks = nn.ModuleList()

        lout = 0
        in_neurons = input_shape
        list_idx = 0
        self.into_convt_idx = 0
        self.into_final_fc_idx = 0
        self.into_mid_fc_idx = 0
        unnested_idx = 0

        # Build convolutional blocks if requested
        if type(self.layers[0]) == list:

            list_idx += 1
            previous = self.layers[0][0]
            for i in range(0, len(self.layers[0])-1):
                self.blocks.append(ConvBlock(previous, 
                                                self.layers[0][i+1], 
                                                kernel_size, 
                                                activations[unnested_idx]))
                
                unnested_idx += 1
                previous = self.layers[0][i+1]
                # Calculate number of flat features
                if i == 0:
                    lout = self.num_flat_features(input_shape, kernel_size)
                else:
                    lout = self.num_flat_features(lout, kernel_size)

            self.flatten1 = nn.Flatten() 

            self.into_mid_fc_idx = len(self.layers[0]) - 1
            conv_out_channels = self.layers[0][-1]
            self.neurons_flattened = lout
            in_neurons = conv_out_channels * lout


        # build fully connected blocks
        previous = in_neurons
        for i in range(list_idx, len(self.layers)):

            if i+1 == len(self.layers) or self.layers[i] == 0 \
                                        or type(self.layers[i+1]) == list:
                
                if type(self.layers[0]) == list and i+1 == len(self.layers):
                    in_neurons = input_shape

                self.blocks.append(LinearBlock(previous, 
                                        in_neurons, activations[unnested_idx]))
                
                unnested_idx += 1
                list_idx += 1
                
                if i+1 == len(self.layers):
                    return
                
                break


            self.blocks.append(LinearBlock(previous, 
                                        self.layers[i], 
                                        activations[unnested_idx]))
            unnested_idx += 1
            list_idx += 1
            previous = self.layers[i]

        self.into_convt_idx = unnested_idx      

        # Build transposed convolutional blocks
        self.into_convt_channels = self.layers[list_idx][0]
        previous = self.layers[list_idx][0]       
        for i in range(0, len(self.layers[list_idx])-1):
            self.blocks.append(ConvTBlock(previous, 
                                            self.layers[list_idx][i+1], 
                                            kernel_size, activations[unnested_idx]))
            unnested_idx += 1
            previous = self.layers[list_idx][i+1]
            # Calculate number of flat features
            if i == 0:
                lout = self.num_flat_features_convt(self.neurons_flattened, 
                                                            kernel_size) 
            else:
                lout = self.num_flat_features_convt(lout, kernel_size)
        
        list_idx += 1
        self.into_final_fc_idx = unnested_idx
        self.flatten2 = nn.Flatten()

        in_neurons =  lout

        # return without final fc layer if not defined
        if unnested_idx + 1  == num_elements(self.layers) - 2 \
            and in_neurons == input_shape:
            return

        if in_neurons == self.layers[list_idx]:
            unnested_idx += 1
            list_idx += 1

        # build final fully connected blocks
        for i in range(list_idx, len(self.layers)):
            self.blocks.append(LinearBlock(in_neurons, 
                                    self.layers[i], activations[unnested_idx]))
            unnested_idx += 1
            in_neurons = self.layers[i]

    def forward(self, x):
        """
        Takes in x, returns it as the model output after being passed 
        through each block.

        Parameters
        ----------
            x : array-like 
                feature vector with dimension [batch_size, input features]  

        Returns
        ----------
            x : array-like 
                feature vector with dimension [batch_size, output features]  
          
        """
        
        for i, block in enumerate(self.blocks):
            

            if i == 0 and type(self.layers[0]) == list and len(x.shape) == 2:
                x = x.reshape(-1, self.layers[0][0], self.layers[-1])

            if i == self.into_mid_fc_idx and self.into_mid_fc_idx != 0:
                x = self.flatten1(x)
            
            if i == self.into_convt_idx and self.into_convt_idx != 0:
                x = x.reshape(-1,self.into_convt_channels,self.neurons_flattened)

            if i == self.into_final_fc_idx and self.into_final_fc_idx != 0:
                x = self.flatten2(x)

            x = block(x)   

            if i == len(self.blocks)-1 and len(x.shape) == 3:
                x = self.flatten2(x) 

        return x


    def num_flat_features(self, input_shape, kernel_size, stride=1, pad=0, 
                                                                dilation=1):

        lout = (input_shape + (2 * pad) - (dilation * (kernel_size - 1)) \
                                                    - 1)// stride + 1
        
        return lout


    def num_flat_features_convt(self, lin, kernel_size, stride=1, pad=0, 
                                                dilation=1, out_pad=0):

        lout = (lin -1)  * stride - 2 * pad  + dilation * (kernel_size - 1) \
                                                    +  out_pad + 1
        
        return lout


                                                                                
def build_model(X, architecture, activations, kernel_size):

    #  use gpu if available
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # get size of one single snapshot
    [rows, columns] = X.shape
    datapnts_at_t = rows

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(datapnts_at_t, architecture, activations, kernel_size) \
                                                        .to(device)

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
            # Reshape mini-batch data to [N, input_shape] matrix and load 
            # it to the active device
            #batch_features = batch_features.view(-1, 1, input_shape).to(device)
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

