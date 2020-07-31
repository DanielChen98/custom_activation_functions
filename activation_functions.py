#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:23:27 2020

@author: shayansadeghieh
"""

# -*- coding: utf-8 -*-
print('running scripts')
import numpy as np
"""Layer definitions."""
class Layer(object):
    """Abstract class defining the interface for a layer."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        raise NotImplementedError()

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
                    Slope known as 'alpha'.
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        
        """
        
        if inputs > 0:
            return inputs
        return self.alpha*(np.exp**inputs - 1)
    
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
        return grads_wrt_outputs if inputs > 0 else grads_wrt_outputs*self.alpha*np.exp(inputs)

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
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        
        """
    
        if inputs > 0:
            return self.scale*inputs
        return self.scale*self.alpha*(np.exp**inputs - 1)
    
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
        return grads_wrt_outputs*self.scale if inputs > 0 else grads_wrt_outputs*self.scale*self.alpha*np.exp(inputs)

class GeluLayer(Layer):
    """Layer implementing a gaussian error linear unit transformation."""
    def fprop(self, inputs):
        """Forward propagates through the gaussian error linear unit layer transformation.
        
        For inputs 'x' and outputs 'y' this corresponds to 
        'y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)'   
        
        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        
        """
        return 0.5*inputs*(1 + np.tanh(np.sqrt(2/np.pi)*(inputs + 0.044715*inputs**3)))
    
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
        
        Derivative of GELU Layer:
        See attached pdf Latex
        
        """
        return grads_wrt_outputs*(((np.tanh((np.sqrt(2)*(0.044715*inputs**3 + inputs))/np.sqrt(np.pi))+((np.sqrt(2)*inputs*(0.134145*inputs**2+1)*((1/np.cosh((np.sqrt(2)*(0.044715*inputs**3+inputs))/np.sqrt(np.pi)))**2))/np.sqrt(np.pi) + 1)))/2)

def IsrlLayer(Layer):
    """Layer implementing an inverse square root linear unit transformation.""" 
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fprop(self, inputs):
        """Forward propagates through the inverse square root linear unit layer transformation.
        
        For inputs 'x' and outputs 'y' this corresponds to 'y = x' if 
        x >= 0 and y = x*(1/sqrt(1+alpha*x^2)) if x > 0, where alpha is a hyperparameter which
        controls the value to which an elu saturates for negative net inputs (see
        https://openreview.net/pdf?id=HkMCybx0-).
        
        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
                    Slope known as 'alpha'. 
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        
        """
        
        if inputs >= 0:
            return inputs
        return inputs*(1/np.sqrt(1+self.alpha*inputs**2))
    
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
        return grads_wrt_outputs if inputs >= 0 else grads_wrt_outputs*(1/np.sqrt(1+self.alpha*inputs**2))**3

print('ran successfully')

