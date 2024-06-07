import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init

class S3Layer(nn.Module):
    def __init__(self, num_segments, shuffle_vector_dim=2):
        super(S3Layer, self).__init__()

        # This is n from the paper
        self.num_segments = int(num_segments)
        self.activation = "relu"

        # Set to He if you want to use He initialisation
        # Otherwise set to None
        self.initialisation_method = "He"

        # shuffle_vector_dim is lamba from the paper
        # This decides how many dimensions a shuffle vector will have
        # For example, if num_segments -> 4
        #   ex 1. shuffle_vector_dim = 1
        #       then shape of shuffle_vector is (4)

        #   ex 2. shuffle_vector_dim = 2
        #       then shape of shuffle_vector is (4,4)

        #   ex 3. shuffle_vector_dim = 3
        #       then shape of shuffle_vector is (4,4,4)

        # The idea was to add some complexity to the shuffle_vector tensor so that it can learn more complex relationships if necessary
        self.shuffle_vector_dim = shuffle_vector_dim

        print(locals())

        # Code to make shuffle vector dimension dynamic
        # I just add num_segments to a tuple as many times as the shuffle_vector_dimension variable is
        # Here the shuffle_vector_shape could be one of the following depending on the value of shuffle vector dim:
        #   (n)
        #   (n,n)
        #   (n,n,n)
        #   (n,n,n,n)
        shuffle_vector_shape = []
        for i in range(0, self.shuffle_vector_dim):
            print(i)
            shuffle_vector_shape.append(self.num_segments)
        shuffle_vector_shape = tuple(shuffle_vector_shape)

        # I create an empty parametric tensor
        # It's empty because I will add weights to it using He initialsation
        self.shuffle_vector = nn.Parameter(torch.empty(shuffle_vector_shape).to("cuda"))

        # Initialise the shuffling parameters
        # This is a design choice, and you can remove the He initialisation and initialise all weights from the same value. 
        # We saw that removing it doesn't affect the training negatively.
        if self.initialisation_method=="He" and shuffle_vector_dim>1:
            init.kaiming_normal_(self.shuffle_vector, mode='fan_out', nonlinearity=self.activation)
            scale_factor = 0.001
            shift_value = 0.01
            self.shuffle_vector.data = self.shuffle_vector.data * scale_factor + shift_value
        # initialise the shuffling parameters manually
        else:
            scale_factor = 0.1  # Adjust the scale factor as needed
            shift_value = 0.5  # Adjust the shift value as needed
            self.shuffle_vector = nn.Parameter(torch.ones(self.num_segments) * scale_factor + shift_value)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        if True:
            init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity=self.activation)

    def forward(self, x):
        # x is of the shape (b, t, c)

        # Total time steps means how many data points are there in the input sequence
        total_time_steps = x.size(1)

        # Now we know the number of steps in the input sequence
        # And we know how many segments to divide them into using num_segments hyperparameter passed to the model
        # So we calculate how many steps should be in each segment
        steps_per_segment = total_time_steps // self.shuffle_vector.size(0)

        # The code below divides the input sequence into n segments and returns a list
        segments = [x[:, i * steps_per_segment: (i + 1) * steps_per_segment, :] for i in range(self.shuffle_vector.size(0))]
        
        # Now the logic I use to re-arrange the segments is explained below using a simple 1 dimensional shuffle vector:
        # 
        #   Let's say the num_segments is 4 
        #   and shuffle_vector for this iteration is [0.01, 0.05, 0.06, 0.005]

        #   Take the index of the largest number and put segment at that index in the first position
        #       Index of largest weight -> 2
        #       Put segment at index 2 in the first position

        #   Then take the index of the second largest number and put the segment at that index in the second position
        #       Index of next largest weight -> 1
        #       Put segment at index 1 in the second position
        
        #   Do this for all segments and at the end
        #   you will have a segment list that is shuffled according to the shuffle vector


        # The above example was for a single dimensional tensor
        # If the shuffle vector is higher dimensional, then take the sum of each row in the last dimension so that it becomes one dimensional in the end
        if len(self.shuffle_vector.shape)>1:
            self.shuffle_vector_sum = self.shuffle_vector.sum(tuple(range(len(self.shuffle_vector.shape)-1)))
        else:
            self.shuffle_vector_sum = self.shuffle_vector
        
        # Now get the list of indices in the descending order of the weight values
        # So if shuffle vector is [0.01, 0.05, 0.06, 0.005]
        # The descending indices are [2,1,0,3]

        # So if shuffle_vector (or shuffle_vector_sum) is [0.05, 0.06, 0.006, 0.0001, 0.005, 0.01]
        # The descending indices are [1, 0, 5, 2, 4, 3]
        sigma = Variable(torch.argsort(self.shuffle_vector_sum, descending=True), requires_grad=False)
        
        # Simply re-arranging the segments using the sigma tensor does not flow gradients through the shuffle vectors
        # So the code below helps in redirecting the gradient flow through it through some hacks

        # Create an intermediate tensor of zeros with the shape (n,n) where n is the number of segments
        omega = Variable(torch.zeros((len(self.shuffle_vector_sum), len(self.shuffle_vector_sum)), device=x.device), requires_grad=False)

        # Now fill in the values in each row according to the descending indices.
        # So if shuffle_vector (or shuffle_vector_sum) is [0.01, 0.05, 0.06, 0.005]
        # The descending indices are [2,1,0,3]

        # The result matrix will be
        # 0,        0,      0.06,   0
        # 0,        0.05,   0,      0
        # 0.01,     0,      0,      0
        # 0,        0,      0,      0.005
        
        for index, i in enumerate(sigma):
            omega[index][i] = self.shuffle_vector_sum[i]

        # The code below will convert the non-zero elements in the result matrix to 1
        non_zero_mask = omega != 0
        scaling_factors = 1.0 / omega[non_zero_mask]
        omega[non_zero_mask] *= scaling_factors

        # The result matrix will be
        # 0,        0,      1,      0
        # 0,        1,      0,      0
        # 1,        0,      0,      0
        # 0,        0,      0,      1

        # Stack the list of segments into a tensor with an extra dimension
        # If input shape ->     (32, 96, 9)
        # Then stack shape->    (32, 24, 9, 4)
        stacked_segments = Variable(torch.stack(segments, dim=-1), requires_grad=False)
        
        # Empty list where the shuffled segments will be stored
        shuffled_segments = []

        # Add an extra dimension to help us with row wise matrix multiplication between result matrix and stacked_segments
        stacked_segments = stacked_segments.unsqueeze(-1).expand(-1, -1, -1, -1, stacked_segments.shape[-1])
        # stack shape->    (32, 24, 9, 4, 4)

        # The idea is that we can treat the 4 segments as a single element, 
        # and then perform dot product of each segment with the corresponding element in one row of result matrix

        # segments -> [S0, S1, S2, S3] (they are all tensors but let's treat them as a single element)
        # 
        # Result Matrix ->
        # [
            # [0,0,1,0],
            # [0,1,0,0],
            # [1,0,0,0],
            # [0,0,0,1],
        # ]

        # So we want to multiply like this:
        #   1. segments * omega[0] (dot product of segments and 0th row of omega)           ->          S0*0 + S1*0 + S2*1 + S3*0 = S2
        #   2. segments * omega[1] (dot product of segments and 1st row of omega)           ->          S0*0 + S1*1 + S2*0 + S3*0 = S1
        #   3. segments * omega[2] (dot product of segments and 2nd row of omega)           ->          S0*1 + S1*0 + S2*0 + S3*0 = S0
        #   4. segments * omega[3] (dot product of segments and 3rd row of omega)           ->          S0*0 + S1*0 + S2*0 + S3*1 = S3

        # What this will do is, for each row it will only retain the segment whose index had a 1 in the result matrix.

        # Dot Product
        ## Multiply the matrices
        multiplication_out = stacked_segments * torch.transpose(omega, 0, 1)
        multiplication_out = torch.transpose(multiplication_out, -2, -1)
        ## Get the sum
        shuffled_segments_stack = multiplication_out.sum(dim=-1)
        
        # Remove extra dimension we had added
        shuffled_segments_list = shuffled_segments_stack.unbind(dim=-1)
        
        # Concatenate the shuffled segments into the original input shape
        # shape of concatenated_segments -> (32, 96, 9)
        concatenated_segments = torch.cat(shuffled_segments_list, dim=1)

        # Now the concatenated_segments tensor has the final shuffled sequence

        # Let's do stitch now

        b,t,c = x.shape

        stacked_shuffle_original = torch.stack((concatenated_segments, x), dim=-1)
        stacked_shuffle_original_reshaped = stacked_shuffle_original.view(b * t * c, 1, 2)

        # This will do a weighted average
        out = self.conv1d(stacked_shuffle_original_reshaped)
        out = out.view(b, t, c)
        # out shape -> (32, 96, 9)
        # out has the same shape as the input, so the rest of the model has no idea we did something ;)
        return out
