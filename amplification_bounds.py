from math import sqrt, log, exp
from scipy.optimize import root_scalar
from scipy.stats import binom
import numpy as np
import math
import scipy.stats as stats
from scipy.fft import fft, ifft
from collections import defaultdict
import cupy as cp

class ShuffleAmplificationBound:
    """Base class for "privacy amplification by shuffling" bounds."""

    def __init__(self, name='BoundBase', tol=None):
        """Parameters:
            name (str): Name of the bound
            tol (float): Error tolerance for optimization routines
        """
        self.name = name
        # Set up a default tolerance for optimization even if none is specified
        if tol is None:
            self.tol_opt = 1e-12
        else:
            self.tol_opt = tol
        # Tolerance for delta must be larger than optimization tolerance
        self.tol_delta = 10*self.tol_opt

    def get_name(self, with_mech=True):
        return self.name

    def get_delta(self, eps, eps0, n):
        """This function returns delta after shuffling for given parameters:
            eps (float): Target epsilon after shuffling
            eps0 (float): Local DP guarantee of the mechanism being shuffled
            n (int): Number of randomizers being shuffled
        """
        raise NotImplementedError

    def threshold_delta(self, delta):
        """Truncates delta to reasonable parameters to avoid numerical artifacts"""
        # The ordering of the arguments is important to make sure NaN's are propagated
        return min(max(delta, self.tol_delta), 1.0)


class Erlingsson(ShuffleAmplificationBound):
    """Implement the bound from Erlignsson et al. [SODA'19]"""

    def __init__(self, name='EFMRTT\'19', tol=None):
        super(Erlingsson, self).__init__(name=name, tol=tol)
        # The constants in the bound are only valid for a certain parameter regime
        self.max_eps0 = 0.5
        self.min_n = 1000
        self.max_delta = 0.01

    def check_ranges(self, eps=None, eps0=None, n=None, delta=None):
        """Check that a set of parameters is within the range of validity of the bound"""
        if eps0 is not None:
            assert eps0 <= self.max_eps0
            if eps is not None:
                assert eps <= eps0
        if n is not None:
            assert n >= self.min_n
        if delta is not None:
            assert delta <= self.max_delta

    def get_delta(self, eps, eps0, n):
        """Implement the bound delta(eps,eps0,n) in [EFMRTT'19]"""
        try:
            self.check_ranges(eps=eps, eps0=eps0, n=n)
            delta = exp(-n * (eps / (12 * eps0))**2)
            self.check_ranges(delta=delta)
        except AssertionError:
            return np.nan

        return self.threshold_delta(delta)

    def get_eps(self, eps0, n, delta):
        """Implement the bound eps(eps0,n,delta) in [EFMRTT'19]"""
        try:
            self.check_ranges(eps0=eps0, n=n, delta=delta)
            eps = 12*eps0*sqrt(log(1/delta)/n)
            self.check_ranges(eps=eps, eps0=eps0)
        except AssertionError:
            return np.nan

        return eps

    def get_eps0(self, eps, n, delta):
        """Implement the bound eps0(eps,n,delta) in [EFMRTT'19]"""
        try:
            self.check_ranges(eps=eps, n=n, delta=delta)
            eps0 = eps/(12*sqrt(log(1/delta)/n))
            self.check_ranges(eps=eps, eps0=eps0)
        except AssertionError:
            return np.nan

        return eps0


class NumericShuffleAmplificationBound(ShuffleAmplificationBound):
    """Base class for amplification bounds that are given in implicit form:
    F(eps,n,mechanism) <= delta
    This class implements the numerics necessary to recover eps and eps0 from implicit bounds.
    """

    def __init__(self, mechanism, name, tol=None):
        """Numeric bounds depend on properties of the mechanism"""
        super(NumericShuffleAmplificationBound, self).__init__(name=name, tol=tol)
        self.mechanism = mechanism

    def get_name(self, with_mech=True):
        if with_mech:
            return '{}, {}'.format(self.name, self.mechanism.get_name())
        return self.name

    def get_delta(self, eps, eps0, n):
        """Getting delta is bound dependent"""
        raise NotImplementedError

    def get_eps(self, eps0, n, delta, min_eps=1e-10):
        """Find the minimum eps giving <= delta"""

        assert eps0 >= min_eps
        # If this assert fails consider decreasing min_eps
        assert self.get_delta(min_eps, eps0, n) >= delta

        def f(x):
            return self.get_delta(x, eps0, n) - delta

        # Use numeric root finding
        sol = root_scalar(f, bracket=[min_eps, eps0], xtol=self.tol_opt)

        assert sol.converged
        eps = sol.root

        return eps

    def get_eps0(self, eps, n, delta, max_eps0=10):
        """Find the maximum eps0 giving <= delta"""

        assert eps <= max_eps0
        # If this assert fails consider increasing max_eps0
        assert self.get_delta(eps, max_eps0, n) >= delta

        def f(x):
            current_delta = self.get_delta(eps, x, n)
            return current_delta - delta

        # Use numeric root finding
        sol = root_scalar(f, bracket=[eps, max_eps0], xtol=self.tol_opt)

        assert sol.converged
        eps0 = sol.root

        return eps0


class Hoeffding(NumericShuffleAmplificationBound):
    """Numeric amplification bound based on Hoeffding's inequality"""

    def __init__(self, mechanism, name='Blanket-hoeffding', tol=None):
        super(Hoeffding, self).__init__(mechanism, name, tol=tol)

    def get_delta(self, eps, eps0, n):

        if eps >= eps0:
            return self.tol_delta

        self.mechanism.set_eps0(eps0)

        gamma_lb, gamma_ub = self.mechanism.get_gamma()
        a = exp(eps) - 1
        b = self.mechanism.get_range_l(eps)

        delta = 1/(gamma_lb*n)
        delta *= b**2 / (4*a)
        delta *= (1 - gamma_lb*(1-exp(-2 * a**2 / b**2)))**n

        return self.threshold_delta(delta)


class BennettExact(NumericShuffleAmplificationBound):
    """Numeric amplification bound based on Bennett's inequality"""

    def __init__(self, mechanism, name='Blanket-bennett', tol=None):
        super(BennettExact, self).__init__(mechanism, name, tol=tol)

    def get_delta(self, eps, eps0, n):

        if eps >= eps0:
            return self.tol_delta

        self.mechanism.set_eps0(eps0)

        gamma_lb, gamma_ub = self.mechanism.get_gamma()
        a = exp(eps) - 1
        b_plus = self.mechanism.get_max_l(eps)
        c = self.mechanism.get_var_l(eps)

        alpha = c / b_plus**2
        beta = a * b_plus / c
        eta = 1.0 / b_plus

        def phi(u):
            phi = (1 + u) * log(1 + u) - u
            if phi < 0:
                # If phi < 0 (due to numerical errors), u should be small
                # enough that we can use the Taylor approximation instead.
                phi = u**2
            return phi

        exp_coef = alpha * phi(beta)
        div_coef = eta * log(1 + beta)

        def expectation_l(m):
            coefs = np.exp(-m * exp_coef) / div_coef
            return coefs

        delta = 1 / (gamma_lb * n)
        expectation_term = binom.expect(expectation_l, args=(n, gamma_lb), lb=1, tolerance=self.tol_opt, maxcount=100000)
        delta *= expectation_term

        return self.threshold_delta(delta)

    
class FFTbound(NumericShuffleAmplificationBound):
    """Compute the optimal amplification bound based on FFT"""

    def __init__(self, mechanism, name='Our FFT', tol=None):
        super(FFTbound, self).__init__(mechanism, name, tol=tol)

    def discretize_distribution(self,support, prob, discretize_delta):
        '''Round x to the nearest larger n*discretize_delta, where n is an integer'''

        grid_dist = defaultdict(float)
        start=1e100
        end=-1e100
        
        for x, p in zip(support, prob):
            x_rounded = math.ceil(x / discretize_delta)
            grid_dist[x_rounded] += p
            start=min(start,x_rounded)
            end=max(end,x_rounded)
        
        length = int(end-start) + 1 
        
        grid_vector = np.zeros(length)
        for x, p in grid_dist.items():
            index = int(x - start)  
            grid_vector[index] += p

        grid_vector=grid_vector/np.sum(grid_vector)
        
        return grid_vector, start

    def fft_compute(self,n,x_original,P_original,discretize_delta):
        '''Compute (1/n)*E[sum_{i=1}^n G_i]_+  via FFT'''
        '''Cupy is used to accelerate the fft'''

        
        grid_vector, start=self.discretize_distribution(x_original,P_original,discretize_delta)

        grid_vector = cp.asarray(grid_vector)
        N = len(grid_vector)
        target_length = 1 << (int(n * (N - 1)).bit_length() ) # FFT will be accelerated when N=2^m, m is an integer
        if N < target_length:
            P_padded = cp.pad(grid_vector, (0, target_length - N), mode='constant')
        else:
            P_padded = grid_vector

        P_fft = cp.fft.fft(P_padded)
        P_fft_n = P_fft ** n

        P_conv_n = cp.fft.ifft(P_fft_n)
        P_conv_n = cp.maximum(P_conv_n, 0) 
        P_conv_n=P_conv_n/ cp.sum(P_conv_n)
        P_conv_n=P_conv_n.real[int(-start)*n:n * (N-1)+1] # remove the negative and the padded portions


        final_support = cp.linspace(0, (N-1+start) * discretize_delta, len(P_conv_n))
        expectation = cp.sum(final_support * P_conv_n)

        ans=expectation
        return ans.item()


    def get_delta(self, eps, eps0, n):
        support_list,prob_list=self.mechanism.get_gparv_distribution(eps)
        discretize_delta=(exp(eps0)-1)/1200
        delta=self.fft_compute(n,support_list,prob_list,discretize_delta)
        return self.threshold_delta(delta)

    def closedformanalysis(self,n, eps0, delta): 
        '''Use the close form of standard clone to determine an upper bound'''
        if eps0 > log(n / (16 * log(4 / delta))):
            return eps0
        else:
            a = 8 * (exp(eps0) * log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
            c = 8 * exp(eps0) / n
            e = log(1 + a + c)
            b = 1 - exp(-eps0)
            d = (1 + exp(-eps0 - e))
            return log(1 + (b / d) * (a + c))
    
    def get_eps(self, eps0, n, delta, min_eps=1e-6):
        """Find the minimum eps giving <= delta"""

        assert eps0 >= min_eps
        # If this assert fails consider decreasing min_eps
        assert self.get_delta(min_eps, eps0, n) >= delta

        epsupper = self.closedformanalysis(n, eps0, delta)
        epslower=min_eps

        while (epsupper-epslower)>1e-5:
            mid=(epslower+epsupper)/2
            delta_new=self.get_delta(mid,eps0,n)
            if delta_new>delta:
                epslower=mid
            else:
                epsupper=mid
        print('bingo!')
        return epsupper
    
class FFTLowerBound(NumericShuffleAmplificationBound):
    """Compute the amplification lower bound based on FFT"""
    '''This method is similar to the FFTBound except the Round method in the discretize_distribution.'''

    def __init__(self, mechanism, name='Our FFT', tol=None):
        super(FFTLowerBound, self).__init__(mechanism, name, tol=tol)

    def discretize_distribution(self,support, prob, discretize_delta):
        '''Round x to the nearest smaller n*discretize_delta, where n is an integer'''

        grid_dist = defaultdict(float)
        start=1e100
        end=-1e100
        
        for x, p in zip(support, prob):
            x_rounded = math.ceil(x / discretize_delta)
            grid_dist[x_rounded] += p
            start=min(start,x_rounded)
            end=max(end,x_rounded)
        length = int(end - start) + 1 
        
        grid_vector = np.zeros(length)
        for x, p in grid_dist.items():
            index = int(x - start)  
            grid_vector[index] += p

        grid_vector=grid_vector/np.sum(grid_vector)
        
        return grid_vector, start

    def fft_compute(self,n,x_original,P_original,discretize_delta):
        '''Compute (1/n)*E[sum_{i=1}^n G_i]_+  via FFT'''
        '''Cupy is used to accelerate the fft'''

        grid_vector, start=self.discretize_distribution(x_original,P_original,discretize_delta)
        grid_vector = cp.asarray(grid_vector)
        N = len(grid_vector)
        target_length = 1 << (int(n * (N - 1)).bit_length() ) # FFT will be accelerated when N=2^m, m is an integer
        if N < target_length:
            P_padded = cp.pad(grid_vector, (0, target_length - N), mode='constant')
        else:
            P_padded = grid_vector

        P_fft = cp.fft.fft(P_padded)
        P_fft_n = P_fft ** n

        P_conv_n = cp.fft.ifft(P_fft_n)
        P_conv_n = cp.maximum(P_conv_n, 0) 
        P_conv_n=P_conv_n/ cp.sum(P_conv_n)
        P_conv_n=P_conv_n.real[int(-start)*n:n * (N-1)+1] # remove the negative and the padded portions


        final_support = cp.linspace(0, (N-1+start) * discretize_delta, len(P_conv_n))
        expectation = cp.sum(final_support * P_conv_n)

        ans=expectation
        return ans.item()


    def get_delta(self, eps, eps0, n):
        support_list,prob_list=self.mechanism.get_gparv_distribution(eps)
        discretize_delta=(exp(eps0)-1)/1200
        delta=self.fft_compute(n,support_list,prob_list,discretize_delta)
        return self.threshold_delta(delta)

    def closedformanalysis(self,n, eps0, delta): 
        '''Use the close form of standard clone to determine an upper bound'''
        if eps0 > log(n / (16 * log(4 / delta))):
            return eps0
        else:
            a = 8 * (exp(eps0) * log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
            c = 8 * exp(eps0) / n
            e = log(1 + a + c)
            b = 1 - exp(-eps0)
            d = (1 + exp(-eps0 - e))
            return log(1 + (b / d) * (a + c))
    
    def get_eps(self, eps0, n, delta, min_eps=1e-6):
        """Find the minimum eps giving <= delta"""

        assert eps0 >= min_eps
        # If this assert fails consider decreasing min_eps
        assert self.get_delta(min_eps, eps0, n) >= delta

        epsupper = self.closedformanalysis(n, eps0, delta)
        epslower=min_eps
        
        while (epsupper-epslower)>1e-5:
            mid=(epslower+epsupper)/2
            delta_new=self.get_delta(mid,eps0,n)
            if delta_new>delta:
                epslower=mid
            else:
                epsupper=mid
        print('bingo!')
        return epsupper


class CloneTheoreticalbound(NumericShuffleAmplificationBound):
    """Implement the theoretical bound from Feldman et al. [FOCS'21]"""

    def __init__(self, mechanism, name='Standard clones-theoretical', tol=None):
        super(CloneTheoreticalbound, self).__init__(mechanism, name, tol=tol)

    def get_eps(self, epsorig, n, delta, min_eps=0.000001):
        assert epsorig >= min_eps

        if epsorig > log(n / (16 * log(4 / delta))):
            # print("This is not a valid parameter regime for this analysis")
            return epsorig
        else:
            a = 8 * (exp(epsorig) * log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
            c = 8 * exp(epsorig) / n
            e = log(1 + a + c)
            b = 1 - exp(-epsorig)
            d = (1 + exp(-epsorig - e))
            return log(1 + (b / d) * (a + c))
    


class CloneEmpiricalBound(NumericShuffleAmplificationBound):
    """Implement the empirical bound from Feldman et al. [FOCS'21]"""

    def __init__(self, mechanism, name='Standard clones-empirical', tol=None):
        super(CloneEmpiricalBound, self).__init__(mechanism, name, tol=tol)
    
    def binarysearch(self,f, delta, num_iterations, epsupper):
        llim = 0
        rlim = epsupper
        for t in range(num_iterations):
            mideps = (rlim + llim) / 2
            delta_for_mideps = f(mideps, delta)
            if delta_for_mideps < delta:
                llim = llim
                rlim = mideps
            else:
                llim = mideps
                rlim = rlim
        return rlim

    def onestep(self,c, eps, eps0, pminusq):
        '''
        onestep computes the e^(eps)-divergence between p=alpha*Bin(c,0.5)+(1-alpha)*(Bin(c,1/2)+1) and q=alpha*(Bin(c,0.5)+1)+(1-alpha)*Bin(c,1/2), where alpha=e^(eps)/(1+e^(eps))
        if pminusq=True then computes D_(e^eps)(p|q), else computes D_(e^eps)(q|p)
        '''
        alpha = math.exp(eps0) / (math.exp(eps0) + 1)
        effeps = math.log(((math.exp(eps) + 1) * alpha - 1) / ((1 + math.exp(eps)) * alpha - math.exp(eps)))
        if pminusq == True:
            beta = 1 / (math.exp(effeps) + 1)
        else:
            beta = 1 / (math.exp(-effeps) + 1)
        cutoff = beta * (c + 1)
        pconditionedonc = (alpha * stats.binom.cdf(cutoff, c, 0.5) + (1 - alpha) * stats.binom.cdf(cutoff - 1, c, 0.5))
        qconditionedonc = ((1 - alpha) * stats.binom.cdf(cutoff, c, 0.5) + alpha * stats.binom.cdf(cutoff - 1, c, 0.5))
        if pminusq == True:
            return (pconditionedonc - math.exp(eps) * qconditionedonc)
        else:
            return ((1 - qconditionedonc) - math.exp(eps) * (1 - pconditionedonc))


    def get_delta(self,eps,eps0,n, deltaupper=1.0, step=1, upperbound = True):
        '''
        Let C=Bin(n-1, e^(-eps0)) and A=Bin(c,1/2) and B=Bin(c,1/2)+1 and alpha=e^(eps0)/(e^(eps0)+1)
        p samples from A w.p. alpha and B otherwise
        q samples from B w.p. alpha and A otherwise
        deltacomp attempts to find the smallest delta such P and Q are (eps,delta)-indistinguishable, or outputs deltaupper if P and Q are not (eps, deltaupper)-indistinguishable.
        If upperbound=True then this produces an upper bound on the true delta (except if it exceeds deltaupper), and if upperbound=False then it produces a lower bound.
        '''
        deltap = 0  # this keeps track of int max{0, p(x)-q(x)} dx
        deltaq = 0  # this keeps track of int max{0, q(x)-p(x)} dx
        probused = 0  # To increase efficiency, we're only to search over a subset of the c values.
        # This will keep track of what probability mass we have covered so far.

        p = math.exp(-eps0)
        expectation = (n-1)*p

        # Now, we are going to iterate over the n/2, n/2-step, n/2+step, n/2-2*steps, ...
        for B in range(1, int(np.ceil(n/step)), 1):
            for s in range(2):
                if s == 0:
                    if B==1:
                        upperc = int(np.ceil(expectation+B*step))  # This is stepping up by "step".
                        lowerc = upperc - step
                    else:
                        upperc = int(np.ceil(expectation + B * step))  # This is stepping up by "step".
                        lowerc = upperc - step + 1
                    if lowerc>n-1:
                        inscope = False
                    else:
                        inscope = True
                        upperc = min(upperc, n-1)
                if s == 1:
                    lowerc = int(np.ceil(expectation-B*step))
                    upperc = lowerc + step - 1
                    if upperc<0:
                        inscope = False
                    else:
                        inscope = True
                        lowerc = max(0, lowerc)

                if inscope == True:
                    cdfinterval = stats.binom.cdf(upperc, n - 1, p) - stats.binom.cdf(lowerc, n - 1, p) + stats.binom.pmf(lowerc, n - 1, p)
                # This is the probability mass in the interval (in Bin(n-1, p))

                    if max(deltap, deltaq) > deltaupper:
                        return deltaupper

                    if 1 - probused < deltap and 1 - probused < deltaq:
                        if upperbound == True:
                            return max(deltap + 1 - probused, deltaq + 1 - probused)
                        else:
                            return max(deltap, deltaq)

                    else:
                        deltap_upperc = self.onestep(upperc, eps, eps0, True)
                        deltap_lowerc = self.onestep(lowerc, eps, eps0, True)
                        deltaq_upperc = self.onestep(upperc, eps, eps0, False)
                        deltaq_lowerc = self.onestep(lowerc, eps, eps0, False)

                        if upperbound == True:
                            # compute the maximum contribution to delta in the segment.
                            # The max occurs at the end points of the interval due to monotonicity
                            deltapadd = max(deltap_upperc, deltap_lowerc)
                            deltaqadd = max(deltaq_upperc, deltaq_upperc)
                        else:
                            deltapadd = min(deltap_upperc, deltap_lowerc)
                            deltaqadd = min(deltaq_upperc, deltaq_lowerc)

                        deltap = deltap + cdfinterval * deltapadd
                        deltaq = deltaq + cdfinterval * deltaqadd

                    probused = probused + cdfinterval  # updates the mass of C covered so far

        return max(deltap, deltaq)
    
    def closedformanalysis(self,n, eps0, delta):
        '''
        Theoretical computation the privacy guarantee of achieved by shuffling n eps0-DP local reports.
        '''
        if eps0 > math.log(n / (16 * math.log(4 / delta))):
            # print("This is not a valid parameter regime for this analysis")
            return eps0
        else:
            a = 8 * (math.exp(eps0) * math.log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
            c = 8 * math.exp(eps0) / n
            e = math.log(1 + a + c)
            b = 1 - math.exp(-eps0)
            d = (1 + math.exp(-eps0 - e))
            return math.log(1 + (b / d) * (a + c))



    # #if UL=1 then produces upper bound, else produces lower bound.
    def get_eps(self, eps0, n, delta, min_eps=0.000001):
        '''
        Empirically computes the privacy guarantee of achieved by shuffling n eps0-DP local reports.
        num_iterations = number of steps of binary search, the larger this is, the more accurate the result
        If upperbound=True then this produces an upper bound on the true shuffled eps, and if upperbound=False then it produces a lower bound.
        '''
        # in order to speed things up a bit, we start the search for epsilon off at the theoretical upper bound.
        if eps0 < math.log(n / (16 * math.log(4 / delta))):
            # checks if this is a valid parameter regime for the theoretical analysis.
            # If yes, uses the theoretical upper bound as a starting point for binary search
            epsupper = self.closedformanalysis(n, eps0, delta)
        else:
            epsupper = eps0
        
        num_iterations=10
        step=8
        upperbound=True

        def deltacompinst(eps, delta):
            return self.get_delta(eps,eps0, n, delta, step, upperbound)

        return self.binarysearch(deltacompinst, delta, num_iterations, epsupper)