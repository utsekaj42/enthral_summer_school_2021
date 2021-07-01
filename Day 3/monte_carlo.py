"""
Module with Monte Carlo methods for uncertainty and  sensitivity analysis using the chaospy package for sampling
"""
import numpy as np
import chaospy as cp
def uq_measures_dummy_func():
    if __name__ == '__main__':
        from sensitivity_examples_nonlinear import generate_distributions
        from sensitivity_examples_nonlinear import linear_model

    # start uq
    # generate the distributions for the problem
    Nrv = 4
    c = 0.5
    zm = np.array([[0., i] for i in range(1, Nrv + 1)])
    wm = np.array([[i * c, i] for i in range(1, Nrv + 1)])
    jpdf = generate_distributions(zm, wm)

    # 1. Generate a set of Xs
    Ns = 20000
    Xs = jpdf.sample(Ns, rule='R').T  # <- transform the sample matrix

    # 2. Evaluate the model
    Zs = Xs[:, :Nrv]
    Ws = Xs[:, Nrv:]
    Ys = linear_model(Ws, Zs)

    # 3. Calculate expectation and variance
    EY = np.mean(Ys)
    VY = np.var(Ys, ddof=1)  # NB: use ddof=1 for unbiased variance estimator, i.e /(Ns - 1)

    print('E(Y): {:2.5f} and  Var(Y): {:2.5f}'.format(EY, VY))
    # end uq


# sample matrices
def generate_sample_matrices_mc(Ns, number_of_parameters, jpdf, sample_method='R'):
    """
    Generate the sample matrices A, B and C_i for Saltelli's algorithm
    Inputs:
    Ns (int): Number of independent samples for each matrices A and B
    jpdf (dist): distribution object with a sample method
    sample_method (str): specify sampling method to use
    Returns A, B, C
    A, B (arrays): samples are rows and parameters are columns
    C (arrays): samples are first dimension, second index indicates which parameter is fixed,
    and parameters values third are indexed by third dimension
    """
    if not jpdf.stochastic_dependent:
        # Fix for Sobol sequence issue where A and B are not
        # independent if formed by dividing sequence in two parts
        # TODO submit pull request for a better way to accomplish this with chaospy
        j2pdf = [eval("cp."+d.__str__()) for d in jpdf] + [eval("cp."+d.__str__())for d in jpdf]
        j2pdf = cp.J(*j2pdf)
        Xtot = j2pdf.sample(Ns, sample_method)
        A = Xtot[0:number_of_parameters].T
        B = Xtot[number_of_parameters:].T
    else:
        raise NotImplementedError('You cannot use this method for dependent distributions')

    C = np.empty((number_of_parameters, Ns, number_of_parameters))
    # create C sample matrices
    for i in range(number_of_parameters):
        C[i, :, :] = B.copy()
        C[i, :, i] = A[:, i].copy()

    return A, B, C



# mc algorithm for variance based sensitivity coefficients
def calculate_sensitivity_indices_mc(y_a, y_b, y_c, main='sobol', total='homma'):
    """
    Sobol's 1993 algorithm for S_m using Monte Carlo integration
    Sobol's 2007 algorithm for S_t using Monte Carlo integration
    Saltelli's 2010 algorithm for estimating S_m and 
    Homma's 1996 algorithm for estimating S_t
    
    Inputs: y_a, y_b (array): first index corresponds to sample second to variables of interest
    y_c (array): first index corresponds conditional index, second to sample and 
        following dimensions to variables of interest
        main (str): specify which method from ('sobol', 'saltelli') to use for S_m
        total (str): specify which method from ('sobol', 'homma') to use for S_t
        
    Returns: s, st
        s (array): first order sensitivities first index corrresponds to input second 
            to variable of interest
        st (array): total sensitivities first index corrresponds to input second 
            to variable of interest
    """
    s_shape = y_c.shape[0:1] + y_c.shape[2:]
    s = np.zeros(s_shape)
    st = np.zeros(s_shape)

    mean = 0.5*(np.mean(y_a,axis=0) + np.mean(y_b,axis=0))
    y_a_center = y_a - mean
    y_b_center = y_b - mean
    f0sq = np.mean(y_a_center,axis=0) * np.mean(y_b_center,axis=0) # 0 when data is centered
    var_est = np.var(y_b, axis=0)
    for i, y_c_i in enumerate(y_c):
        y_c_i_center = y_c_i - mean
        if main=='sobol':
            s[i] = (np.mean(y_a_center*y_c_i_center, axis=0)-f0sq)/var_est 
        elif main=='saltelli':
            s[i] = np.mean(y_a_center*(y_c_i_center - y_b_center), axis=0)/var_est 
        else:
            raise ValueError('Unknown method main="%s"' % main)

        if total=='homma':
            st[i] = 1 - (np.mean(y_c_i_center*y_b_center, axis=0) - f0sq)/var_est 
        elif total=='sobol':
            st[i] = np.mean(y_b_center*(y_b_center-y_c_i_center), axis=0)/var_est
        else:
            raise ValueError('Unknown method total="%s"' % total)
    return s, st


# end mc algorithm for variance based sensitivity coefficients
