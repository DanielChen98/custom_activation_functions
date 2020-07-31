#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:23:27 2020

@author: shayansadeghieh
"""

# -*- coding: utf-8 -*-
"""Layer definitions."""


class EluLayer(Layer):
    """Layer implementing an exponential linear unit transformation."""
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fprop(self, inputs):
    """Forward propagates through the exponential layer transformation.
    
    For inputs 'x' and outputs 'y' this corresponds to 'y = x' if 
    x > 0 and y = alpha*(e^x-1) if x <= 0 where alpha is an elu hyperparameter which
    controls the value to which an elu saturates for negative net inputs (see
    https://arxiv.org/pdf/1511.07289.pdf).
    
    Args:
        inputs: Array of layer inputs of shape (batch_size, input_dim).
                Slope known as 'alpha' which is usually b/w 0.1-0.3.
        outputs: Array of layer outputs of shape (batch_size, output_dim).
    
    """
    
    if x > 0:
        return x
    return self.alpha*(np.exp**x - 1)
    
    def bprop(self, inputs, outputs, grads_wrt_outputs):
    """Back propagates gradients through the layer.
    
    Given gradients with respect to the outputs of the layer, it calculates the
    gradients with respect to the layer inputs.
    
    Args:
        inputs: Array of layer inputs of shape (batch_size, input_dim).
        outputs: Array of layer outputs calculated in forward pass of shape
        (batch_size, output_dim).
        grads_wrt_outputs: Array of gradients with respect to the layer
        outputs of shape (batch_size, output_dim).
    
    Returns:
        Array of gradients with respect to the layer inputs of shape
        (batch_size, input_dim).
    
    """
    return grads_wrt_outputs if x > 0 else grads_wrt_outputs*self.alpha*np.exp(x)

class SeluLayer(Layer):
    """Layer implementing a scaled exponential linear unit transformation."""
    def __init__(self, alpha, scale):
        self.alpha = alpha
        self.scale = scale
    
    def fprop(self, inputs):
    """Forward propagates through the scaled exponential layer transformation.
    
    For inputs 'x' and outputs 'y' this corresponds to 'y = scale*x' if 
    x > 0 and y = scale*alpha*(e^x-1) if x <= 0 where 'alpha' is an elu hyperparameter which
    controls the value to which an elu saturates for negative net inputs (see
    https://arxiv.org/pdf/1511.07289.pdf) and 'scale' is a hyperparameter which 
    ensures the elu can self-normalize (see https://arxiv.org/pdf/1706.02515.pdf)
    
    Args:
        inputs: Array of layer inputs of shape (batch_size, input_dim).
                Slope known as 'alpha' which is usually b/w 0.1-0.3.
                Scale which allows for self-normalization.
        outputs: Array of layer outputs of shape (batch_size, output_dim).
    
    """
    
    if x > 0:
        return self.scale*x
    return self.scale*self.alpha*(np.exp**x - 1)
    
    def bprop(self, inputs, outputs, grads_wrt_outputs):
    """Back propagates gradients through the layer.
    
    Given gradients with respect to the outputs of the layer, it calculates the
    gradients with respect to the layer inputs.
    
    Args:
        inputs: Array of layer inputs of shape (batch_size, input_dim).
        'Alpha' which is usually b/w 0.1 and 0.3.
        'Scale' which is a hyerparamter to ensure self-normalization.
        outputs: Array of layer outputs calculated in forward pass of shape
        (batch_size, output_dim).
        grads_wrt_outputs: Array of gradients with respect to the layer
        outputs of shape (batch_size, output_dim).
    
    Returns:
        Array of gradients with respect to the layer inputs of shape
        (batch_size, input_dim).
    
    """
    return grads_wrt_outputs*self.scale if x > 0 else grads_wrt_outputs*self.scale*self.alpha*np.exp(x)

    
        
    
    
    
    

