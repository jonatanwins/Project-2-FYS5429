from typing import Dict, Union
from jax import Array


# PARAMS OBJECT STRUCTURE

""" 
Example of params object as it should be passed in model.apply(params, input)
where model is a nn.Module. Where Array(shape) is a jax array with the given shape.

{'params': 
    {'decoder': 
        {'Dense_0': {'bias': Array(20,), 'kernel': Array(2, 20)},
        'Dense_1': {'bias': Array(40,), 'kernel': Array(20, 40)}, 
        'Dense_2': {'bias': Array(60,), 'kernel': Array(40, 60)}, 
        'Dense_3': {'bias': Array(128,), 'kernel': Array(60, 128)}}, 
    'encoder': 
        {'Dense_0': {'bias': Array(60,), 'kernel': Array(128, 60)}, 
        'Dense_1': {'bias': Array(40,), 'kernel': Array(60, 40)}, 
        'Dense_2': {'bias': Array(20,), 'kernel': Array(40, 20)}, 
        'Dense_3': {'bias': Array(2,), 'kernel': Array(20, 2)}}, 
    'sindy_coefficients': Array(3, 2)}}
"""

# Type for the innermost dictionary (holding the 'bias' and 'kernel' tensors).
LayerParams = Dict[str, Array]

# Type for unnested nn.module parameters, such as decoder and encoder.
ModelParams = Dict[str, LayerParams]


# Type for the 'params' key in the 'params' object.
ModelLayers = Dict[str, Union[ModelParams, Array]]

# Nested dictionary structure for the 'params' object, required for flax nn.module apply method.
NestedModelLayers = Dict[str, ModelLayers]
