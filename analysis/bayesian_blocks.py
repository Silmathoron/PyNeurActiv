#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the PyNeurActiv project, which aims at providing tools
# to study and model the activity of neuronal cultures.
# Copyright (C) 2017 SENeC Initiative
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This particular file is taken from the `atroML <http://www.astroml.org/>`
# project and is provided under a BSD license.
#
# Modified by Tanguy Fardet (6/26/2017): added min/max_width keywords

"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle et al 2012 [1]_

References
----------
.. [1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
"""

import numpy as np


class FitnessFunc(object):

    '''
    Base class for fitness functions

    Each fitness function class has the following:
    - fitness(...) : compute fitness function.
       Arguments accepted by fitness must be among [T_k, N_k, a_k, b_k, c_k]
    - prior(N, Ntot) : compute prior on N given a total number of points Ntot
    '''

    def __init__(self, p0=0.05, gamma=None, **kwargs):
        self.p0 = p0
        self.gamma = gamma

    def validate_input(self, t, x, sigma):
        '''
        Check that input is valid
        '''
        pass

    def fitness(**kwargs):
        raise NotImplementedError()

    def prior(self, N, Ntot):
        if self.gamma is None:
            return self.p0_prior(N, Ntot)
        else:
            return self.gamma_prior(N, Ntot)

    def p0_prior(self, N, Ntot):
        # eq. 21 from Scargle 2012
        return 4 - np.log(73.53 * self.p0 * (N ** -0.478))

    def gamma_prior(self, N, Ntot):
        '''
        Basic prior, parametrized by gamma (eq. 3 in Scargle 2012)
        '''
        if self.gamma == 1:
            return 0
        else:
            return (np.log(1 - self.gamma)
                    - np.log(1 - self.gamma ** (Ntot + 1))
                    + N * np.log(self.gamma))

    # the fitness_args property will return the list of arguments accepted by
    # the method fitness(). This allows more efficient computation below.
    @property
    def args(self):
        try:
            # Python 2
            return self.fitness.func_code.co_varnames[1:]
        except AttributeError:
            return self.fitness.__code__.co_varnames[1:]


class Events(FitnessFunc):

    """
    Fitness for binned or unbinned events

    Parameters
    ----------
    p0 : float
        False alarm probability, used to compute the prior on N
        (see eq. 21 of Scargle 2012).  Default prior is for p0 = 0.
    gamma : float or None
        If specified, then use this gamma to compute the general prior form,
        p ~ gamma^N.  If gamma is specified, p0 is ignored.
    """

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        pos = (N_k != 0) * (T_k != 0)
        res = np.zeros(len(N_k))
        res[pos] = N_k[pos] * (np.log(N_k[pos]) - np.log(T_k[pos]))
        return res

    def prior(self, N, Ntot):
        if self.gamma is not None:
            return self.gamma_prior(N, Ntot)
        else:
            # eq. 21 from Scargle 2012
            return 4 - np.log(73.53 * self.p0 * (N ** -0.478))


class RegularEvents(FitnessFunc):

    """
    Fitness for regular events

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    gamma : float
        specifies the prior on the number of bins: p ~ gamma^N
    """

    def __init__(self, dt, p0=0.05, gamma=None):
        self.dt = dt
        self.p0 = p0
        self.gamma = gamma

    def validate_input(self, t, x, sigma):
        unique_x = np.unique(x)
        if list(unique_x) not in ([0], [1], [0, 1]):
            raise ValueError("Regular events must have only 0 and 1 in x")

    def fitness(self, T_k, N_k):
        # Eq. 75 of Scargle 2012
        M_k = T_k / self.dt
        N_over_M = N_k * 1. / M_k

        eps = 1E-8
        if np.any(N_over_M > 1 + eps):
            import warnings
            warnings.warn('regular events: N/M > 1.  '
                          'Is the time step correct?')

        one_m_NM = 1 - N_over_M
        N_over_M[N_over_M <= 0] = 1
        one_m_NM[one_m_NM <= 0] = 1

        return N_k * np.log(N_over_M) + (M_k - N_k) * np.log(one_m_NM)


class PointMeasures(FitnessFunc):

    """
    Fitness for point measures

    Parameters
    ----------
    gamma : float
        specifies the prior on the number of bins: p ~ gamma^N
        if gamma is not specified, then a prior based on simulations
        will be used (see sec 3.3 of Scargle 2012)
    """

    def __init__(self, p0=None, gamma=None):
        self.p0 = p0
        self.gamma = gamma

    def fitness(self, a_k, b_k):
        # eq. 41 from Scargle 2012
        return (b_k * b_k) / (4 * a_k)

    def prior(self, N, Ntot):
        if self.gamma is not None:
            return self.gamma_prior(N, Ntot)
        elif self.p0 is not None:
            return self.p0_prior(N, Ntot)
        else:
            # eq. at end of sec 3.3 in Scargle 2012
            return 1.32 + 0.577 * np.log10(N)


def bayesian_blocks(t, data=None, sigma=None, fitness='events', min_width=0.,
                    max_width=np.inf, **kwargs):
    """
    Bayesian Blocks Implementation

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2012 [1]_

    Parameters
    ----------
    t : array_like
        Times (one dimensional, length N)
    data : array_like (optional)
        Data values
    sigma : array_like or float (optional)
        Data errors
    fitness : str or object
        the fitness function to use.
        If a string, the following options are supported:

        - 'events' : binned or unbinned event data
            extra arguments are `p0`, which gives the false alarm probability
            to compute the prior, or `gamma` which gives the slope of the
            prior on the number of bins.
            For this method, additional `min_width` and `max_width` keywords
            can be provided to prevent excessively large or small bins.
        - 'regular_events' : non-overlapping events measured at multiples
            of a fundamental tick rate, `dt`, which must be specified as an
            additional argument.  The prior can be specified through `gamma`,
            which gives the slope of the prior on the number of bins.
        - 'measures' : fitness for a measured sequence with Gaussian errors
            The prior can be specified using `gamma`, which gives the slope
            of the prior on the number of bins.  If `gamma` is not specified,
            then a simulation-derived prior will be used.

        Alternatively, the fitness can be a user-specified object of
        type derived from the FitnessFunc class.
    min_width : float, optional (default: 0.)
        Minimum width accepted for a bin.
    max_width : float, optional (default: infinity)
        Maximum acceptable width for a bin.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) bin edges

    Examples
    --------
    Event data:

    >>> t = np.random.normal(size=100)
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Regular event data:

    >>> dt = 0.01
    >>> t = dt * np.arange(1000)
    >>> data = np.zeros(len(t))
    >>> data[np.random.randint(0, len(t), len(t) / 10)] = 1
    >>> bins = bayesian_blocks(t, fitness='regular_events', dt=dt, gamma=0.9)

    Measured point data with errors:

    >>> t = 100 * np.random.random(100)
    >>> data = np.exp(-0.5 * (t - 50) ** 2)
    >>> sigma = 0.1
    >>> data_obs = np.random.normal(data, sigma)
    >>> bins = bayesian_blocks(t, fitness='measures')

    References
    ----------
    .. [1] Scargle, J `et al.` (2012)
           http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    See Also
    --------
    :func:`astroML.plotting.hist` : histogram plotting function which can make
        use of bayesian blocks.
    """
    # validate array input
    t = np.asarray(t, dtype=float)
    if data is not None:
        data = np.asarray(data)
    if sigma is not None:
        sigma = np.asarray(sigma)

    # verify the fitness function
    if fitness == 'events':
        if data is not None and np.any(data % 1 > 0):
            raise ValueError(
                "`data` must be integer counts for fitness='events'")
        fitfunc = Events(**kwargs)
    elif fitness == 'regular_events':
        if data is not None and (np.any(data % 1 > 0) or np.any(data > 1)):
            raise ValueError(
                "`data` must be 0 or 1 for fitness='regular_events'")
        fitfunc = RegularEvents(**kwargs)
    elif fitness == 'measures':
        if data is None:
            raise ValueError("`data` must be specified for fitness='measures'")
        fitfunc = PointMeasures(**kwargs)
    else:
        if not (hasattr(fitness, 'args') and
                hasattr(fitness, 'fitness') and
                hasattr(fitness, 'prior')):
            raise ValueError("fitness not understood")
        fitfunc = fitness

    # find unique values of t
    t = np.array(t, dtype=float)
    assert t.ndim == 1
    unq_t, unq_ind, unq_inv = np.unique(t, return_index=True,
                                        return_inverse=True)

    # if data is not specified, data will be counts at each time
    if data is None:
        if sigma is not None:
            raise ValueError(
                "If `sigma` is specified, `data` must be specified")

        if len(unq_t) == len(t):
            data = np.ones_like(t)
        else:
            data = np.bincount(unq_inv)

        t = unq_t
        sigma = 1

    # if data is specified, then we need to sort t and data together
    else:
        data = np.asarray(data)

        if len(t) != len(data):
            raise ValueError("Size of `t` and `data` does not match")

        if len(unq_t) != len(t):
            raise ValueError("Repeated values in t not supported when "
                             "`data` is specified")
        t = unq_t
        data = data[unq_ind]

    # verify the given sigma value
    N = t.size
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape not in [(), (1,), (N,)]:
            raise ValueError('`sigma` does not match the shape of `data`')
    else:
        sigma = 1

    # validate the input
    fitfunc.validate_input(t, data, sigma)

    # compute values needed for computation, below
    if 'a_k' in fitfunc.args:
        ak_raw = np.ones_like(data) / sigma / sigma
    if 'b_k' in fitfunc.args:
        bk_raw = data / sigma / sigma
    if 'c_k' in fitfunc.args:
        ck_raw = data * data / sigma / sigma

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)
        kwds = {}

        # T_k: width/duration of each block
        if 'T_k' in fitfunc.args:
            kwds['T_k'] = np.clip(
                block_length[:R + 1] - block_length[R + 1],
                min_width, max_width)

        # N_k: number of elements in each block
        if 'N_k' in fitfunc.args:
            kwds['N_k'] = np.cumsum(data[:R + 1][::-1])[::-1]

        # a_k: eq. 31
        if 'a_k' in fitfunc.args:
            kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]

        # b_k: eq. 32
        if 'b_k' in fitfunc.args:
            kwds['b_k'] = - np.cumsum(bk_raw[:R + 1][::-1])[::-1]

        # c_k: eq. 33
        if 'c_k' in fitfunc.args:
            kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = fitfunc.fitness(**kwds)

        A_R = fit_vec - fitfunc.prior(R + 1, N)
        A_R[1:] += best[:R]

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]

    #-----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]
