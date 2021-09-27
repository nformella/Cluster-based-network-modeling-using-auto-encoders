'''
Modules and functions to build and train an autoencoder.

Classes:

    AE(AE, int, list, list)
    ConvBlock()
    ConvTBlock()
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
#import torchvision      # for testing
from prettytable import PrettyTable


class ConvBlock(nn.Module):
    """    
    Class adds a block with a convolutional layer followed by an 
    activation function to the neural network.

    ...

    Attributes
    ----------
    in_channels : int
        input channels
    kernel_size: int or tupel
        kernel size
    out_channels : int
        output channels
    activation : function
        nonlinearity/activation function

    Methods
    -------
    forward(x):
        Passes input through the convolutional block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                                                        activation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_activation = activation

    def forward(self, x):
        return self.conv_activation(self.conv(x))



class ConvTBlock(nn.Module):
    """    
    Class adds a block with a transposed convolutional layer 
    followed by an activation function to the neural network.

    ...

    Attributes
    ----------
    in_channels : int
        input channels
    kernel_size: int or tupel
        kernel size
    out_nchannels : int
        output channels
    activation : function
        nonlinearity/activation function

    Methods
    -------
    forward(x):
        Passes input through the convolutional block.
    """

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

        if self.activation == 'lin':
            return self.linear(x)

        return self.activation(self.linear(x))



class NNDecoder(nn.Module):
    def __init__(self, input_shape, architecture, activations, 
                                                kernel_size,
                                                list_idx,
                                                unnested_idx,
                                                in_neurons,
                                                neurons_flattened,
                                                latent_features,
                                                multi_decoder=False):

        super(NNDecoder, self).__init__()

        self.layers = architecture.copy()
        self.layers.append(input_shape)
        self.neurons_flattened = neurons_flattened
        self.into_final_fc_idx = 0

        self.into_convt_idx = 0

        fewest_neurons_per_layer_idx = unnested_idx
    
        # Construct each layer followed by an activation function and
        # store it in self.blocks
        self.blocks = nn.ModuleList()

        self.count = 0

        list_idx, unnested_idx = self.build_fc_blocks(unnested_idx, 
                                                        in_neurons,
                                                        latent_features, 
                                                        list_idx, 
                                                        input_shape,
                                                        activations,
                                                        multi_decoder)
        if unnested_idx == 0:
            return

        self.count += 1
        self.into_convt_idx = unnested_idx - fewest_neurons_per_layer_idx

        # Build transposed convolutional blocks
        unnested_idx, lout = self.build_convt_blocks(list_idx, kernel_size, 
                                                            activations, 
                                                            unnested_idx)
        list_idx += 1
        self.into_final_fc_idx = unnested_idx - fewest_neurons_per_layer_idx
        self.flatten2 = nn.Flatten() 

        in_neurons =  lout

        # Return without final fc layer if it is not defined. Just use the 
        # flattened vector in that case.
        if unnested_idx == num_elements(self.layers) - 1 \
            and in_neurons == input_shape:
            return

        # Only add a single fully connected block if the size of the 
        # final user defined layer is not the size of the flattened
        # vector.
        if in_neurons == self.layers[list_idx]:
            list_idx += 1 
        
        # build final fully connected blocks
        self.build_fc_blocks(unnested_idx, in_neurons, latent_features,
                                                        list_idx, 
                                                        input_shape,
                                                        activations)

        

    def build_fc_blocks(self, unnested_idx, in_neurons, latent_features,
                                                             list_idx, 
                                                            input_shape,
                                                            activations,
                                                    multi_decoder=False):

        lower = list_idx

        previous = latent_features

        # If each latent feature has its own decoder
        if self.count == 0 and multi_decoder == True:
            previous = 1
        elif self.count == 1:
            previous = self.layers[list_idx-1]
        
        for i in range(lower, len(self.layers)):

            if i+1 == len(self.layers) or self.layers[i] == 0 \
                                or type(self.layers[i+1]) == list:

                # If final AE layer is reached, this will reset the 
                # out_features to the input_shape
                if type(self.layers[0]) == list and \
                                            i+1 == len(self.layers):
                    in_neurons = input_shape

                self.blocks.append(LinearBlock(previous, in_neurons, 
                                            activations[unnested_idx]))
                
                
                unnested_idx += 1
                list_idx += 1
   
                if i+1 == len(self.layers):
                    return 0, 0
                
                break
                

            self.blocks.append(LinearBlock(previous, self.layers[i], 
                                        activations[unnested_idx]))
        
            unnested_idx += 1
            list_idx += 1

            previous = self.layers[i]

        return list_idx, unnested_idx


    def build_convt_blocks(self, list_idx, kernel_size, activations, 
                                                        unnested_idx):

        # Flattened vector is reshaped into a convt box without adding
        # an additional block. This is accounted for by adding the "1" 
        # to the unnested_idx
        unnested_idx += 1 
        self.into_convt_channels = self.layers[list_idx][0]
        previous = self.layers[list_idx][0]       
        for i in range(0, len(self.layers[list_idx])-1):
            self.blocks.append(ConvTBlock(previous, 
                                        self.layers[list_idx][i+1], 
                                        kernel_size, 
                                        activations[unnested_idx]))
        
            unnested_idx += 1
            previous = self.layers[list_idx][i+1]
            # Calculate number of flat features
            if i == 0:
                lout = self.num_flat_features_convt(self.neurons_flattened, 
                                                            kernel_size) 
            else:
                lout = self.num_flat_features_convt(lout, kernel_size)

        return unnested_idx, lout



    def num_flat_features_convt(self, lin, kernel_size, stride=1, pad=0, 
                                                dilation=1, out_pad=0):
        """
        Takes in a height of data points, kernel_size, stride, pad, dilation 
        and returns the calculated height after a transposed convolutional 
        block.

        Parameters
        ----------
            
            dilation: int or tuple, optional
                Spacing between kernel elements. Default: 1
            input_shape : int
                Height of data points
            kernel_size : int or tuple
                Size of the convolving kernel. Default: 1
            out_pad: int or tuple, optional
                Zero-padding added to both sides of the input. Default: 0
            stride: int or tuple, optional
                Stride of the convolution. Default: 1  

        Returns
        ----------
            lout : int
                Height of data points  
          
        """

        lout = (lin -1)  * stride - 2 * pad  + dilation * (kernel_size - 1) \
                                                    +  out_pad + 1
        
        return lout  


    def forward(self, x):
        """
        Takes in x, returns it as the model output after being passed 
        through each block of the decoder.

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
            

            if i == self.into_convt_idx and self.into_convt_idx != 0:
                x = x.reshape(-1,self.into_convt_channels,self.neurons_flattened)

            if i == self.into_final_fc_idx and self.into_final_fc_idx != 0:
                x = self.flatten2(x)
        
            x = block(x)   

            if i == len(self.blocks)-1 and len(x.shape) == 3:
                x = self.flatten2(x) 

        return x


class NNEncoder(nn.Module):
    def __init__(self, input_shape, architecture, activations, 
                                                kernel_size=50):

        super(NNEncoder, self).__init__()

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

        # Construct each layer followed by an activation function and
        # store it in self.blocks
        self.blocks = nn.ModuleList()

        self.in_neurons = input_shape
        self.into_mid_fc_idx = 0
        self.into_convt_idx = 0
        self.into_final_fc_idx = 0
        self.neurons_flattened = 0
        self.fewest_neurons_per_layer_idx = 0
        
        self.list_idx = 0
        self.unnested_idx = 0

        if type(self.layers[0]) == list:  
            self.build_conv_blocks(kernel_size, activations, input_shape)
            self.list_idx += 1

        # build fully connected blocks and find the index of the latent 
        # vector (smallest number of neurons)
        self.fewest_neurons_per_layer_idx = self.list_idx-1
        self.build_fc_blocks(input_shape,activations)
        self.latent_features = self.layers[self.fewest_neurons_per_layer_idx]



    def build_conv_blocks(self, kernel_size, activations, input_shape):

        previous = self.layers[0][0]
        for i in range(0, len(self.layers[0])-1):
            self.blocks.append(ConvBlock(previous, 
                                    self.layers[0][i+1], 
                                    kernel_size, 
                                    activations[self.unnested_idx]))
                
            self.unnested_idx += 1
            previous = self.layers[0][i+1]
            # Calculate number of flat features
            if i == 0:
                lout = self.num_flat_features_conv(input_shape, 
                                                        kernel_size)
            else:
                lout = self.num_flat_features_conv(lout, kernel_size)

        self.flatten1 = nn.Flatten() 

        self.into_mid_fc_idx = len(self.layers[0]) - 1
        conv_out_channels = self.layers[0][-1]
        self.neurons_flattened = lout
        self.in_neurons = conv_out_channels * lout



    def num_flat_features_conv(self, input_shape, kernel_size, stride=1, pad=0, 
                                                                dilation=1):
        """
        Takes in the input_shape, kernel_size, stride, pad, dilation and 
        returns the calculated height after a convolutional block.

        Parameters
        ----------
            
            dilation: int or tuple, optional
                Spacing between kernel elements. Default: 1
            input_shape : int
                Height of data points
            kernel_size : int or tuple
                Size of the convolving kernel.
            pad: int or tuple, optional
                Zero-padding added to both sides of the input. Default: 0
            stride: int or tuple, optional
                Stride of the convolution. Default: 1  

        Returns
        ----------
            lout : int
                Height of data points  
          
        """

        lout = (input_shape + (2 * pad) - (dilation * (kernel_size - 1)) \
                                                    - 1)// stride + 1
        
        return lout   



    def build_fc_blocks(self, input_shape, activations):

        lower = self.list_idx
        previous = self.in_neurons
        for i in range(lower, len(self.layers)):

            if i+1 == len(self.layers) or self.layers[i] == 0 \
                                or type(self.layers[i+1]) == list:

                # update index of the latent feature vector
                if self.layers[i] <= previous and self.layers[i] != 0:
                    self.fewest_neurons_per_layer_idx = self.unnested_idx
                
                return


            # update index of the latent feature vector in mid fc blocks
            if self.layers[i] <= previous:
                
                self.blocks.append(LinearBlock(previous, self.layers[i], 
                                        activations[self.unnested_idx]))

                self.unnested_idx += 1
                self.list_idx += 1
            
            elif self.layers[i] > previous:
                return

            previous = self.layers[i]
            self.fewest_neurons_per_layer_idx += 1



    def get_list_idx(self):
        return self.list_idx



    def get_unnested_idx(self):
        return self.unnested_idx



    def get_in_neurons(self):
        return self.in_neurons


    
    def forward(self, x):
        """
        Takes in x, returns it as the model output after being passed 
        through each block of the encoder.

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

            x = block(x)
            
        return x


class AE(nn.Module):
    def __init__(self, input_shape, architecture, activations, 
                                                kernel_size,
                                                multi_decoder=True):

        super(AE, self).__init__()

        self.encoder = NNEncoder(input_shape, architecture, 
                                                activations, 
                                                kernel_size)
        in_neurons = self.encoder.in_neurons
        list_idx = self.encoder.list_idx
        unnested_idx = self.encoder.unnested_idx
        neurons_flattened = self.encoder.neurons_flattened
        self.multi_decoder = multi_decoder
        
        self.decoders = []
        
        if self.multi_decoder == True:
            for i in range(0,self.encoder.latent_features):

                instance = 'self.decoder' + str(i+1)
                exec(instance + '=  NNDecoder(input_shape, architecture, \
                                                        activations, \
                                                        kernel_size,\
                                                        list_idx,\
                                                        unnested_idx,\
                                                        in_neurons,\
                                                        neurons_flattened, \
                                            self.encoder.latent_features, \
                                                        multi_decoder)')
                
                exec('self.decoders.append(' + instance + ')')
        else:
            self.decoder = NNDecoder(input_shape, architecture, 
                                                        activations, 
                                                        kernel_size,
                                                        list_idx,
                                                        unnested_idx,
                                                        in_neurons,
                                                        neurons_flattened,
                                               self.encoder.latent_features,
                                                        multi_decoder)
            self.decoders.append(self.decoder)




    def forward(self, x):

        code = self.encoder(x)

        if self.multi_decoder == True:
            neuron_list = []
            dec_outs = []
            z = 0
            for i in range(0,len(code[0])):

                neuron = code[:, i]
                neuron = neuron.reshape(len(neuron), 1)
                dec_out = self.decoders[i](neuron)

                neuron_list.append(neuron)
                dec_outs.append(dec_out)  
                z += dec_out
        
            return z, code, dec_outs

        z = self.decoders[0](code)
        dec_outs = [z]

        return z, code, dec_outs



# class currently not in use
class CompleteAE(nn.Module):
    def __init__(self, input_shape, architecture, activations, 
                                                kernel_size=50):
        """
        Create an autoencoder object derived from torch.nn.Module.

        Parameters
        ----------
            input_shape : int
                input features
            architecture : list
                List with network layers
            activations : list
                List with nonlinearity/activation functions 
            kernel_size : int
                Size of the convolving kernel. Default: 50

        """

        super(CompleteAE, self).__init__()

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

        # Construct each layer followed by an activation function and
        # store it in self.blocks
        self.blocks = nn.ModuleList()

        in_neurons = input_shape
        self.into_mid_fc_idx = 0
        self.into_convt_idx = 0
        self.into_final_fc_idx = 0
        
        list_idx = 0
        unnested_idx = 0

        if type(self.layers[0]) == list:  
            unnested_idx, in_neurons = self.build_conv_blocks(kernel_size, 
                                                            activations, 
                                                            unnested_idx,
                                                            input_shape)
            list_idx += 1

        # build fully connected blocks and find the index of the latent 
        # vector (smallest number of neurons)
        self.count = 0
        self.fewest_neurons_per_layer_idx = unnested_idx
        list_idx, unnested_idx = self.build_fc_blocks(unnested_idx, in_neurons, 
                                                                list_idx, 
                                                                input_shape,
                                                                activations)
        if unnested_idx == 0:
            return

        # remember if build_fc_blocks has been called
        self.count += 1

        self.into_convt_idx = unnested_idx

        # Build transposed convolutional blocks
        unnested_idx, lout = self.build_convt_blocks(list_idx, kernel_size, 
                                                            activations, 
                                                            unnested_idx)
        list_idx += 1
        self.into_final_fc_idx = unnested_idx
        self.flatten2 = nn.Flatten() 

        in_neurons =  lout

        # Return without final fc layer if it is not defined. Just use the 
        # flattened vector in that case.
        if unnested_idx == num_elements(self.layers) - 1 \
            and in_neurons == input_shape:
            return

        # Only add a single fully connected block if the size of the 
        # final user defined layer is not the size of the flattened
        # vector.
        if in_neurons == self.layers[list_idx]:
            list_idx += 1 
        
        # build final fully connected blocks
        self.build_fc_blocks(unnested_idx, in_neurons, list_idx, 
                                                        input_shape,
                                                        activations)



    def build_conv_blocks(self, kernel_size, activations, unnested_idx,
                                                        input_shape):

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
                lout = self.num_flat_features_conv(input_shape, 
                                                        kernel_size)
            else:
                lout = self.num_flat_features_conv(lout, kernel_size)

        self.flatten1 = nn.Flatten() 

        self.into_mid_fc_idx = len(self.layers[0]) - 1
        conv_out_channels = self.layers[0][-1]
        self.neurons_flattened = lout
        in_neurons = conv_out_channels * lout

        return unnested_idx, in_neurons



    def num_flat_features_conv(self, input_shape, kernel_size, stride=1, pad=0, 
                                                                dilation=1):
        """
        Takes in the input_shape, kernel_size, stride, pad, dilation and 
        returns the calculated height after a convolutional block.

        Parameters
        ----------
            
            dilation: int or tuple, optional
                Spacing between kernel elements. Default: 1
            input_shape : int
                Height of data points
            kernel_size : int or tuple
                Size of the convolving kernel.
            pad: int or tuple, optional
                Zero-padding added to both sides of the input. Default: 0
            stride: int or tuple, optional
                Stride of the convolution. Default: 1  

        Returns
        ----------
            lout : int
                Height of data points  
          
        """

        lout = (input_shape + (2 * pad) - (dilation * (kernel_size - 1)) \
                                                    - 1)// stride + 1
        
        return lout



    def build_convt_blocks(self, list_idx, kernel_size, activations, 
                                                        unnested_idx):

        # flattened vector is reshaped into a convt box without adding
        # an additional block. This is accounted for by adding the "1" 
        # to the unnested_idx
        unnested_idx += 1 
        self.into_convt_channels = self.layers[list_idx][0]
        previous = self.layers[list_idx][0]       
        for i in range(0, len(self.layers[list_idx])-1):
            self.blocks.append(ConvTBlock(previous, 
                                        self.layers[list_idx][i+1], 
                                        kernel_size, 
                                        activations[unnested_idx]))
        
            unnested_idx += 1
            previous = self.layers[list_idx][i+1]
            # Calculate number of flat features
            if i == 0:
                lout = self.num_flat_features_convt(self.neurons_flattened, 
                                                            kernel_size) 
            else:
                lout = self.num_flat_features_convt(lout, kernel_size)

        return unnested_idx, lout



    def num_flat_features_convt(self, lin, kernel_size, stride=1, pad=0, 
                                                dilation=1, out_pad=0):
        """
        Takes in a height of data points, kernel_size, stride, pad, dilation 
        and returns the calculated height after a transposed convolutional 
        block.

        Parameters
        ----------
            
            dilation: int or tuple, optional
                Spacing between kernel elements. Default: 1
            input_shape : int
                Height of data points
            kernel_size : int or tuple
                Size of the convolving kernel. Default: 1
            out_pad: int or tuple, optional
                Zero-padding added to both sides of the input. Default: 0
            stride: int or tuple, optional
                Stride of the convolution. Default: 1  

        Returns
        ----------
            lout : int
                Height of data points  
          
        """

        lout = (lin -1)  * stride - 2 * pad  + dilation * (kernel_size - 1) \
                                                    +  out_pad + 1
        
        return lout



    def build_fc_blocks(self, unnested_idx, in_neurons, list_idx, 
                                                        input_shape,
                                                         activations):


        lower = list_idx
        previous = in_neurons
        for i in range(lower, len(self.layers)):

            if i+1 == len(self.layers) or self.layers[i] == 0 \
                                or type(self.layers[i+1]) == list:

                # if final AE layer is reached, this will reset the 
                # out_features to the input_shape
                if type(self.layers[0]) == list and \
                                            i+1 == len(self.layers):
                    in_neurons = input_shape

                self.blocks.append(LinearBlock(previous, in_neurons, 
                                            activations[unnested_idx]))
                
                
                unnested_idx += 1
                list_idx += 1

                # update index of the latent feature vector
                if self.count == 0:
                    if self.layers[i] <= previous and self.layers[i] != 0:
                        self.fewest_neurons_per_layer_idx = unnested_idx-1
                    
                if i+1 == len(self.layers):
                    return 0, 0
                
                break
                

            self.blocks.append(LinearBlock(previous, self.layers[i], 
                                        activations[unnested_idx]))
        
            unnested_idx += 1
            list_idx += 1

            # update index of the latent feature vector in mid fc blocks
            if self.count == 0:
                if self.layers[i] <= previous and self.layers[i] != 0:
                    self.fewest_neurons_per_layer_idx = unnested_idx-1

            previous = self.layers[i]

        return list_idx, unnested_idx



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
            
            if i == self.fewest_neurons_per_layer_idx:
                code = block(x)

            if i == self.into_convt_idx and self.into_convt_idx != 0:
                x = x.reshape(-1,self.into_convt_channels,self.neurons_flattened)

            if i == self.into_final_fc_idx and self.into_final_fc_idx != 0:
                x = self.flatten2(x)
        
            x = block(x)   

            if i == len(self.blocks)-1 and len(x.shape) == 3:
                x = self.flatten2(x) 

        return x, code



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    return total_params, table



def build_model(X, architecture, activations, kernel_size=50, 
                                                multi_decoder=False):
    """
    Creates a model from the `AE` autoencoder class and loads it to the 
    specified device, either gpu or cpu.
        
        Parameters
        ----------
            
            
            activations: list
                Storing information about the activation functions
            architecture: list
                Information about the number of filters and neurons
            kernel_size : int or tuple, optional
                Size of the convolving kernel. Default: 50
            X: tensor
                Data matrix containing the snapshots [input_features, 
                                                            snapshots] 

        Returns
        ----------
            
            model : AE
                Autoencoder class
            device : AE
                Device that the autoencoder class is loaded to  
          
        """

    #  use gpu if available
    #device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    device = pt.device("cpu")

    # get size of one single snapshot
    [rows, columns] = X.shape
    datapnts_at_t = rows

    # MNIST testing
    #datapnts_at_t = 784  # MNIST test

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(datapnts_at_t, architecture, activations, kernel_size, 
                                                        multi_decoder) \
                                                        .to(device)

    return model, device


def prep_training(X, model, learning_rate, batch_size=1, 
                                                data_type="1d_function",
                                                targets=None):
    """
    Prepares the training process by creating an optimizer object, 
    defining the training loss and loading the training data.
        
        Parameters
        ----------
            
            batch_size: int
                Batch size for training. Default: 1
            learning_rate: float
                Dictates the learning rate during training
            model: AE
                Autoencoder class defining the models being trained
            X: tensor
                Data matrix containing the snapshots [input_features, 
                                                    number of snapshots] 
            targets: tensor, optional
                Data matrix containing the snapshots at time t + dt
                [input_features, number of snapshots]. Default: None (no 
                time mapping)

        Returns
        ----------
            
            criterion : criterion
                Applied criterion in optimaization process
            optimizer : optimizer
                Applied optimizer            
            train_loader : DataLoader
                Training data    
        """
   
    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()

    train_dataset = []
    # load train data
    if data_type == "1d_function":
        if targets != None:
            for i in range(len(X.real.T)-1):
                train_dataset.append([X.real.T[i], targets.real.T[i]])
        else:
            train_dataset = X.real.T
    elif data_type == "openFoam":
        if targets != None:
            for i in range(len(X.T)-1):
                train_dataset.append([X.T[i], targets.T[i]])
        else:
            train_dataset = X.T


    # -----------------testing with MNIST--------------
    #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    #train_dataset = torchvision.datasets.MNIST(
    #        root="~/torch_datasets", train=True, transform=transform, download=True
    #        )
    # -------------------------------------------------

    train_loader = pt.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False)

    return optimizer, criterion, train_loader



def check_early_stopping(loss_values, epoch, no_change_for = 1000):
    """Takes in a list of loss values and if the last no_change_for
    entries of that list are the same, returns True."""
    if epoch > no_change_for:
        return len(set(loss_values[epoch-no_change_for:-1])) <= 1



def train_ae(model, epochs, train_loader, input_shape, optimizer, 
                                criterion, device, print_reconLoss=True,
                                                    time_mapping=False):
    """
    Trains the autoencoder for a specified number of epochs. Therefore, 
    mini-batch data is loaded to the active device, a reconstruction is
    calculated and backpropagation is performed based on the output and
    chosen criterion. Each training loss per epoch is stored and returned
    in a list.
        
        Parameters
        ----------
            
            criterion: criterion
                Criterion being used to calculate the loss, e.g., MSE
            device: device
                Determines if CPU or GPU is being used   
            epochs: int
                Defines the number of epochs that the model is trained
            input_shape: int
                Given input features   
            model: AE
                Autoencoder class defining the model being trained
            optimizer: optimizer
                perform parameter update based on current gradients
            print_reconloss: Boolean
                If true the training loss per epoch is printed in real time.
                Default: True
            time_mapping: Boolean
                If true, time mapping is enabled. Default: False

        Returns
        ----------
            
            loss_values : list
                Contains a history of the calculated epoch training loss.  
        """

    loss_values = []
    for epoch in range(epochs):
        # MNIST testing
        #input_shape = 784           

        loss = 0
        for batch_features in train_loader:
            # Reshape mini-batch data to [N, input_shape] matrix and load 
            # it to the active device
            if time_mapping == True:
                inputs, targets = batch_features
                inputs = inputs.view(-1, input_shape).to(device)
            else:
                inputs = batch_features.view(-1, input_shape).to(device)
                targets = batch_features.view(-1, input_shape).to(device)

            # Reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes 
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs, code, dec_out_list = model(inputs)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, targets)
            
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
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, 
                                                            epochs, loss))

        loss_values.append(loss)

        if check_early_stopping(loss_values, epoch, 1000) == True:
            print("No improvement->Stopping further training!")
            break
    
    return loss_values, code, dec_out_list

