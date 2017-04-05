#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tools to compute the properties of the activity """

import numpy as np
from scipy.special import lambertw


def out_of_eq_aeif(params):
    '''
    Check whether aEIF neuronal parameters lead to a stable equilibrium.
    
    Args:
        params (dict): Dictionnary of the neuron's parameters with the keys
            formatted as returned by ``nest.GetStatus(aeif_neuron)``. It must
            contain at least "E_L", "I_e", "a", and "tau_w".
    
    Returns: 
        out_of_eq (bool): ``True`` if the neuron is out of equilibrium;
            ``False`` if a fixed point exists.
    '''
    A = params['E_L'] + params['I_e'] / (1 + params['a'])
    aPlusOne = 1 + params['a']
    has_FP = aPlusOne - np.exp(1 + A) > 0.
    V0minus = np.real(A - lambertw(- np.exp(A) / aPlusOne, 0))
    stable_FP = np.exp(V0minus) - 1. < min(aPlusOne, 1./ params['tau_w'])
    out_of_eq = not (has_FP and stable_FP)
    return out_of_eq


def Vprime_aeif(V, w, EL, Ie, Is):
    '''
    Returns the derivative of the membrane potential for an aEIF neuron.
    
    Args:
        V (double): membrane potential.
        w (double): adaptation variable.
        EL (double): resting potential.
        Ie (double): constant stimulation current.
        Is (double): synaptic current.
    
    Returns:
        -(V-EL) + np.exp(V) + Ie + Is - w
    '''
    return -(V-EL) + np.exp(V) + Ie + Is - w


def inv_Vprime_aeif(V, w, EL, Ie, Is):
    '''
    Inverse of Vprime_aeif.
    
    See Also:
        :func:`Vprime_aeif`
    '''
    return 1. / Vprime_aeif(V, w, EL, Ie, Is)
