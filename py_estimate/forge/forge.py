r"""

==============
Data container
==============

.. moduleauthor:: Antonia Mey <antonia.mey@fu-berlin.de>, Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>

"""

import numpy as np
from ..errors import ExpressionError

class Forge(object):
    r"""
    Parameters
    ----------
    trajs : list of dictionaries
        each dictionary contains the following entries:
        'm' markov sequence in a 1-D numpy array of integers
        't' thermodynamic sequence in a 1-D numpy array of integers
        'b' reduced bias energy sequences in a 2-D numpy array of floats
    b_K_i : numpy.ndarray( shape=(T,M), dtype=numpy.float64 ) (optional)
        reduced bias energies at the T thermodynamic and M discrete Markov states
    kT_K : numpy.ndarray( shape=(T), dtype=numpy.float64 ) (optional)
        array of reduced temperatures of each thermodynamic state K
    kT_target : int (optional)
        integer of the thermodynamic state K of the kT_K array at which estimators
        observables should be estimated
    verbose : boolean (default=False)
        be loud and noisy
    Notes
    -----
    I convert/process the list of trajectory dictionaries into data types that
    are need for pytram estimators.
    """
    def __init__(self, trajs, b_K_i=None, kT_K=None, kT_target=None, verbose=False):
        
        self.trajs = trajs
        self._n_therm_states = None
        self._n_markov_states = None
        self._N_K_i = None
        self._N_K = None
        self._M_x = None
        self._T_x = None
        self._b_K_x = None
        self._M_K_x = None
        self._b_IK_x = None

        self.b_K_i = b_K_i
        self.kT_K = kT_K
        self.kT_target = kT_target
        self.verbose = verbose
        if (kT_K is not None) and (kT_target is None):
            self.kT_target = 0

    ############################################################################
    #
    #   n_markov_states / n_therm_states getters
    #
    ############################################################################

    @property
    def n_markov_states(self):
        if self._n_markov_states is None:
            if self.verbose:
                print("# Counting Markov states")
            self._n_markov_states = 0
            for traj in self.trajs:
                max_state = np.max(traj['m'])
                if max_state > self._n_markov_states:
                    self._n_markov_states = max_state
            self._n_markov_states += 1
            if self.verbose:
                print("# ... found %d Markov states" % self._n_markov_states)
        return self._n_markov_states

    @property
    def n_therm_states(self):
        if self._n_therm_states is None:
            if self.verbose:
                print("# Counting thermodynamic states")
            self._n_therm_states = 0
            for traj in self.trajs:
                max_state = np.max(traj['t'])
                if max_state > self._n_therm_states:
                    self._n_therm_states = max_state
            self._n_therm_states += 1
            if self.verbose:
                print("# ... found %d thermodynamic states" % self._n_therm_states)
        return self._n_therm_states

    ############################################################################
    #
    #   N_K_i / N_K getters
    #
    ############################################################################

    @property   
    def N_K_i(self):
        if self._N_K_i is None:
            if self.verbose:
                print("# Counting visited Markov states")
            self._N_K_i = np.zeros(
                shape=(self.n_therm_states, self.n_markov_states), dtype=np.intc)
            for traj in self.trajs:
                for K in range(self.n_therm_states):
                    inc_K = (traj['t'] == K)
                    for i in range(self.n_markov_states):
                        inc_i = (traj['m'][inc_K] == i)
                        self._N_K_i[K, i] += inc_i.sum()
            if self.verbose:
                print("# ... done")
        return self._N_K_i

    @property
    def N_K(self):
        if self._N_K is None:
            self._N_K = self.N_K_i.sum(axis=1)
        return self._N_K.astype(np.intc)

    ############################################################################
    #
    #   M_x / T_x getters
    #
    ############################################################################

    @property
    def M_x(self):
        if self._M_x is None:
            if self.verbose:
                print("# Copying Markov state sequences")
            self._M_x = np.zeros(shape=(self.N_K.sum(),), dtype=np.intc)
            a = 0
            for traj in self.trajs:
                b = a + traj['m'].shape[0]
                self._M_x[a:b] = traj['m'][:]
                a = b
            if self.verbose:
                print("# ... done")
        return self._M_x

    @property
    def T_x(self):
        if self._T_x is None:
            if self.verbose:
                print("# Copying thermodynamic state sequences")
            self._T_x = np.zeros(shape=(self.N_K.sum(),), dtype=np.intc)
            a = 0
            for traj in self.trajs:
                b = a + traj['t'].shape[0]
                self._T_x[a:b] = traj['t'][:]
                a = b
            if self.verbose:
                print("# ... done")
        return self._T_x

    @property
    def b_K_x(self):
        if self._b_K_x is None:
            if self.verbose:
                print("# Copying bias energy sequences")
            if not self.kT_K is None:
                self._gen_b_K_x_from_kT_K()
            else:
                self._gen_b_K_x()
            if self.verbose:
                print("# ... done")
        return self._b_K_x


    def get_C_K_ij( self, lag, sliding_window=True ):
        r"""
        Parameters
        ----------
        lag : int
            lagtime tau, at which the countmatrix should be evaluated
        sliding_window : boolean (default=True)
            lag is applied by mean of a sliding window or skipping data entries.
        Returns
        -------
        C_K_ij : numpy.ndarray(shape=(T,M,M))
            count matrices C_ij at each termodynamic state K
        """
        C_K_ij = np.zeros(
            shape=(self.n_therm_states, self.n_markov_states, self.n_markov_states),
            dtype=np.intc)
        for traj in self.trajs:
            t = 0
            while t < traj['m'].shape[0]-lag:
                K = traj['t'][t]
                if np.all(traj['t'][t:t+lag+1] == K):
                    C_K_ij[K, traj['m'][t], traj['m'][t+lag]] += 1
                if sliding_window:
                    t += 1
                else:
                    t += lag
        return C_K_ij

    ############################################################################
    #
    #   b_K_x getter and helper functions
    #
    ############################################################################


    def _gen_b_K_x(self):
        self._b_K_x = np.zeros(shape=(self.n_therm_states, self.N_K.sum()), dtype=np.float64)
        a = 0
        for traj in self.trajs:
            if traj['b'].shape[1] == 1:
                raise ExpressionError(
                    "b_K_x",
                    "Trajectory with single energy columns detected - use kT file and kT target")
            if traj['b'].shape[1] != self.n_therm_states:
                raise ExpressionError(
                    "b_K_x",
                    "Trajectory with wrong number of energy columns detected (%d!=%d)" \
                    % (traj['b'].shape[1], self.n_therm_states))
            b = a + traj['b'].shape[0]
            self._b_K_x[:, a:b] = traj['b'][:, :].transpose().copy()
            a = b

    def _gen_b_K_x_from_kT_K(self):
        b_x = np.zeros(shape=(self.N_K.sum(),), dtype=np.float64)
        a = 0
        for traj in self.trajs:
            b = a + traj['b'].shape[0]
            b_x[a:b] = traj['b'][:, 0]
            a = b
        for K in range(self.kT_K.shape[0]):
            b_x[(self.T_x == K)] *= self.kT_K[K]
        self._b_K_x = np.zeros(shape=(self.kT_K.shape[0], self.N_K.sum()), dtype=np.float64)
        for K in range(self.kT_K.shape[0]):
            self._b_K_x[K, :] = (1.0/self.kT_K[K] - 1.0/self.kT_K[self.kT_target]) * b_x[:]


    @property
    def M_K_x( self ):
        if self._M_K_x is None:
            if self.verbose:
                print("# Copying Markov state sequences")
            self._M_K_x = np.zeros( shape=(self.n_therm_states,self.N_K.max()), dtype=np.intc )
            t_K = np.zeros( shape=(self.n_therm_states,), dtype=np.intc )
            for traj in self.trajs:
                # SPEEDUP POSSIBLE!
                for t in range( traj['t'].shape[0] ):
                    K = traj['t'][t]
                    self._M_K_x[K,t_K[K]] = traj['m'][t]
                    t_K[K] += 1
            if self.verbose:
                print("# ... done")
        return self._M_K_x

    @property
    def b_IK_x( self ):
        if self._b_IK_x is None:
            if self.verbose:
                print("# Copying bias energy sequences")
            self._b_IK_x = np.zeros( shape=(self.n_therm_states,self.n_therm_states,self.N_K.max()), dtype=np.intc )
            if not self.kT_K is None:
                self.gen_b_IK_x_from_kT_K()
            else:
                self.gen_b_IK_x()
            if self.verbose:
                print("# ... done")
        return self._b_IK_x

    def gen_b_IK_x_from_kT_K( self ):
        t_K = np.zeros( shape=(self.n_therm_states,), dtype=np.intc )
        for traj in self.trajs:
            # SPEEDUP POSSIBLE!
            for t in range( traj['t'].shape[0] ):
                K = traj['t'][t]
                for I in range( self.n_therm_states ):
                    self._b_IK_x[I,K,t_K[K]] = self.kT_K[K] * ( 1.0/self.kT_K[I] - 1.0/self.kT_K[self.kT_target] ) * traj['b'][t]
                t_K[K] += 1

    def gen_b_IK_x( self ):
        t_K = np.zeros( shape=(self.n_therm_states,), dtype=np.intc )
        for traj in self.trajs:
            if traj['u'].shape[1] == 1:
                raise ExpressionError(
                        "b_IK_x",
                        "Trajectory with single energy columns detected - use kT file and kT target"
                    )
            if traj['u'].shape[1] != self.n_therm_states:
                raise ExpressionError(
                        "b_IK_x",
                        "Trajectory with wrong number of energy columns detected (%d!=%d)" \
                        % ( traj['u'].shape[1], self.n_therm_states )
                    )
            # SPEEDUP POSSIBLE!
            for t in range( traj['t'].shape[0] ):
                K = traj['t'][t]
                for I in range( self.n_therm_states ):
                    self._b_IK_x[I,K,t_K[K]] = traj['b'][I,t]
                t_K[K] += 1