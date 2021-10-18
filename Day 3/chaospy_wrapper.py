import chaospy as cp
import numpy as np
from itertools import combinations

def generate_basis(max_order, dists):
    poly, norms = cp.generate_expansion(max_order, dists, retall=True)
    #TODO: Why return squared norms? When are they useful?
    basis = dict(poly=poly, norms=np.sqrt(norms), dists=dists)
    return basis


def fit_regression(basis, samples, evals):
    poly = basis['poly']
    approx, coeffs = cp.fit_regression(poly, samples, evals, retall=1) 
    expansion = dict(approx=approx, 
                     coeffs=coeffs, 
                     basis=basis)
    return expansion

def fit_quadrature(basis, nodes, weights, evals):
    poly = basis['poly']
    approx, coeffs = cp.fit_quadrature(poly, nodes, weights, evals, retall=1) 
    expansion = dict(approx=approx, 
                     coeffs=coeffs, 
                     basis=basis)
    return expansion


def parse_var_index(var):
    """ Get the index from the string
    Input: var (str): a string symbolic variable that terminates with an integer e.g. q0 or q11
    Returns: var_idx (int) the trailing integer of the string"""
    var_idx = 0
    for place, char in enumerate(reversed(var)):
        try:
            var_idx += int(char)*10**place
        except ValueError:
            return var_idx


def create_variable_masks(expansion):
    dim = len(expansion['basis']['dists'])
    poly = expansion['basis']['poly']
    names_dict = {name:idx for idx, name in enumerate(poly.names)} # TODO: confirm this should be the canonical ordering for the distribution as well?

    var_sel = np.zeros((dim, len(poly)), bool)
    main_sel = np.zeros((dim, len(poly)), bool)
    for base_idx, pol in enumerate(poly):
        nvars = len(pol.names)
        if nvars == 1: # The constant polynomial says it has a variable q0 so we need to catch this
            var_name = pol.names[0]
            if var_name in str(pol):
                var_idx = names_dict[var_name]
                main_sel[var_idx, base_idx] = True
                var_sel[var_idx, base_idx] = True
            else:
                mean_idx = base_idx
        else:
            for var_name in pol.names:
                var_idx = names_dict[var_name]
                var_sel[var_idx, base_idx] = True
    return mean_idx, main_sel, var_sel
                

def calc_sensitivity_indices(expansion, order):
    dim = len(expansion['basis']['dists'])
    poly = expansion['basis']['poly']
    norms = expansion['basis']['norms']
    uhat = expansion['coeffs']
    if len(uhat.shape)>1:
        norms = norms[:,np.newaxis]

    mean_idx, main_sel, var_sel =  create_variable_masks(expansion)
    variance_sel = np.ones(uhat.shape[0], dtype=bool)
    variance_sel[mean_idx] = False
    mean = np.sum(uhat[mean_idx]*norms[mean_idx], axis=0)
    variance = np.sum((uhat[variance_sel]*norms[variance_sel])**2, axis=0)

    main = dict()
    total = dict()
    subsets = combinations(range(dim), order)
    for subset in subsets:
        subset_mask = list(subset) # Need a non-tuple to index numpy array
        complement = []
        for x in range(dim):
            if x in subset:
                pass
            else:
                complement.append(x)
        
        subset_main = np.logical_and(np.all(var_sel[subset_mask], axis=0),
                                     np.logical_not(np.any(var_sel[complement], axis=0)))
        
        subset_total = np.any(var_sel[subset_mask], axis=0)
        main[subset] = np.sum((uhat[subset_main]*norms[subset_main])**2, axis=0)/variance
        total[subset] = np.sum((uhat[subset_total]*norms[subset_total])**2, axis=0)/variance
    return main, total


def calc_descriptives(expansion):
    dim = len(expansion['basis']['dists'])
    poly = expansion['basis']['poly']
    norms = expansion['basis']['norms']
    uhat = expansion['coeffs']
    mean_idx, main_sel, var_sel =  create_variable_masks(expansion)

    #TODO necessary to implement for higher dim outputs?
    output_shape = uhat.shape[1:]
    if len(output_shape) > 0:
        norms = norms[:,np.newaxis]
    main = np.zeros((dim, *output_shape))
    total = np.zeros((dim, *output_shape))   

    variance_sel = np.ones(len(uhat), dtype=bool)
    variance_sel[mean_idx] = False
    mean = np.sum(norms[mean_idx]*uhat[mean_idx:mean_idx+1], axis=0)
    variance = np.sum((norms[variance_sel]*uhat[variance_sel])**2, axis=0)
    for var in range(dim):
        main[var] = np.sum((norms[main_sel[var]]*uhat[main_sel[var]])**2, axis=0)
        total[var] = np.sum((norms[var_sel[var]]*uhat[var_sel[var]])**2, axis=0)

    main = main/variance
    total = total/variance

    stats = dict(mean=mean, 
                 variance=variance, 
                 std=np.sqrt(variance),
                 sens_m=main,
                 sens_t=total)
    return stats


def E(expansion, dists):
    """ TODO: Is this desired? """
    stats = calc_descriptives(expansion)
    return stats['mean']


def Var(expansion, dists):
    """ TODO: Is this desired? """
    stats = calc_descriptives(expansion)
    return stats['variance']


def Std(expansion, dists):
    """ TODO: Is this desired? """
    stats = calc_descriptives(expansion)
    return stats['std']


def Sens_m(expansion, dists):
    """ TODO: Is this desired? """
    stats = calc_descriptives(expansion)
    return stats['sens_m']


def Perc(expansion, percentiles, dists):
    prediction_interval = cp.Perc(expansion['approx'], percentiles , dists)
    return prediction_interval


def Sens_t(expansion, dists):
    stats = calc_descriptives(expansion)
    return stats['sens_t']


def Sens_m2(expansion, dists):
    main, _ = calc_sensitivity_indices(expansion, 2)
    uhat = expansion['coeffs']
    output_shape = uhat.shape[1:]
    dim = len(expansion['basis']['dists'])
    s_shape = (dim, dim) + output_shape
    sens = np.zeros(s_shape)
    for key, value in main.items():
        sens[key[0],key[1]] = value
        sens[key[1],key[0]] = value
    return sens


def test_ishigami():
    # http://www.sfu.ca/~ssurjano/ishigami.html
    # Sobol and Levitan 1999 use a=7 and b=0.05 which works well
    # http://www.andreasaltelli.eu/file/repository/Sobol_Levitan_1999.pdf
    # For a=7 and b=0.1(Marrel et al 2009 Gaussian Processes) seems to be much
    # more challenging for accurate total order indices
    a = 7.
    b = 0.05 #5 
    #1
    #b = 0.10
    def wrapper(z):
        return np.sin(z[0]) + a*np.sin(z[1])**2 + b*z[2]**4*np.sin(z[0])

    D = a**2./8 + b*np.pi**4./5 + b**2*np.pi**8./18 + 1./2
    D1 = b*np.pi**4./5 + b**2*np.pi**8./50. + 1./2
    D2 = a**2/8.
    D3 = 0

    D12  = 0
    D13  = b**2. * np.pi**8 / 18 - b**2*np.pi**8./50.
    D23  = 0
    D123 = 0

    input_names = ["z1", "z2", "z3"]
    s_true = [D1/D, D2/D, D3/D]
    s_t_true = [(D1+D13)/D, D2/D, D13/D]
    
    jpdf = cp.Iid(cp.Uniform(-np.pi, np.pi),3)
    
    polynomial_order = 4 #12 #2 #4
    basis = generate_basis(polynomial_order, jpdf)
    # 2. generate samples with random sampling
    Ns_pc = 4*len(basis['poly'])
    samples_pc = jpdf.sample(size=Ns_pc, rule='S')
    # 3. evaluate the model, to do so transpose samples and hash input data
    #transposed_samples = samples_pc.transpose()
    model_evaluations = wrapper(samples_pc)
    model_evaluations = model_evaluations
    # 4. calculate generalized polynomial chaos expression
    gpce_regression = fit_regression(basis, samples_pc, model_evaluations)
    stats = calc_descriptives(gpce_regression)
    Spc = stats['sens_m']
    Stpc = stats['sens_t']
    print('Wrapper')
    print('mean', stats['mean'], )
    print('variance', stats['variance'], D)
    print("--------------------------------------------------------------------")
    print(Spc)
    print(s_true)
    print("--------------------------------------------------------------------")
    print(Stpc)
    print(s_t_true)
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    S2_dict, _ = calc_sensitivity_indices(gpce_regression, 2)
    print(S2_dict)
    print((0,2), D13/D) 

    
    ## 5. get sensitivity indices
    Spc = stats['sens_m']
    Stpc = stats['sens_t']
    print("chaospy native")
    print("--------------------------------------------------------------------")
    print(Spc)
    print(s_true)
    print(cp.Sens_m(gpce_regression["approx"], jpdf))
    print("--------------------------------------------------------------------")
    print(Stpc)
    print(s_t_true)
    print(cp.Sens_t(gpce_regression["approx"], jpdf))
    print("--------------------------------------------------------------------")
    S2 = cp.Sens_m2(gpce_regression["approx"], jpdf) # second order indices with gpc
    print(S2)
    print((0,2), D13/D, S2[0,2]) 


def test_high_dims(dims=10):
    """ TODO: Write test to confirm functionality for ndarray quantities of interest"""
    w = (np.arange(dims)+1)/dims
    s_true = w**2/np.sum(w**2)
    s_true = np.array((s_true, s_true[::-1])).T
    s_t_true = w**2/np.sum(w**2)
    s_t_true = np.array((s_t_true, s_t_true[::-1])).T
    #w.append(1-np.sum(w))
    def wrapper(z):
        """ wrapper for samples from chaospy, i.e. first index (axis=0) iterates over variables"""
        ret_val = np.array((w@z, w[::-1]@z)).T
        return ret_val

    jpdf = cp.Iid(cp.Uniform(0, 1), dims)
    mean_true = np.sum(w)/2 
    variance_true = np.sum(w**2)/12 

    polynomial_order = 2 # Should be exact with order 1, but good to test with quadractic
    basis = generate_basis(polynomial_order, jpdf)
    # 2. generate samples with Sobol sampling
    Ns_pc = 4*len(basis['poly'])
    samples_pc = jpdf.sample(size=Ns_pc, rule='S')
    # 3. evaluate the model, to do so transpose samples and hash input data
    model_evaluations = wrapper(samples_pc)
    print(model_evaluations.shape)
    # 4. calculate generalized polynomial chaos expression
    gpce_regression = fit_regression(basis, samples_pc, model_evaluations)
    stats = calc_descriptives(gpce_regression)
    print("mean: --------------------------------------------------------------------")
    print(stats['mean'])
    print(mean_true)
    print("Variance: --------------------------------------------------------------------")
    print(stats['variance'])
    print(variance_true)
    
    Spc = stats['sens_m']
    Stpc = stats['sens_t']
    print(Spc.shape, s_true.shape)
    print("S_i--------------------------------------------------------------------")
    for idx in range(dims):
        print(s_true[idx, 0], Spc[idx, 0], s_true[idx,1], Spc[idx,1])
    print("S_t,i--------------------------------------------------------------------")
    for idx in range(dims):
        print(s_t_true[idx, 0], Stpc[idx, 0], s_t_true[idx,1], Stpc[idx,1])
    
def test_array_output():
    """ TODO: Write test to confirm functionality for ndarray quantities of interest"""
    w = [0.25, 0.4]
    w.append(1-np.sum(w))
    def wrapper(z):
        """ wrapper for samples from chaospy, i.e. first index (axis=0) iterates over variables"""
        c1 = np.sum(z, axis=0)
        c2 = np.product(z, axis=0)
        c3 = np.sqrt(w[0])*z[0] + np.sqrt(w[1])*z[1] + np.sqrt(w[2])*z[2]
        return np.swapaxes([c1, c2, c3], 0, -1)

    jpdf = cp.Iid(cp.Uniform(-np.pi, np.pi),3)
    mean_true = [0,0,0]
    variance_true = [3*(2*np.pi)**2/12, None, (2*np.pi)**2/12]

    s_true = np.array([[1/3, 0, w[0]],
                  [1/3, 0, w[1]],
                  [1/3, 0, w[2]]])

    s_t_true = np.array([ [1/3, 1, w[0]],
                  [1/3, 1, w[1]],
                  [1/3, 1, w[2]]])

    polynomial_order = 3 # If order=len(jpdf) should be exact
    basis = generate_basis(polynomial_order, jpdf)
    # 2. generate samples with random sampling
    Ns_pc = 4*len(basis['poly'])
    samples_pc = jpdf.sample(size=Ns_pc, rule='S')
    # 3. evaluate the model, to do so transpose samples and hash input data
    #transposed_samples = samples_pc.transpose()
    model_evaluations = wrapper(samples_pc)
    # 4. calculate generalized polynomial chaos expression
    gpce_regression = fit_regression(basis, samples_pc, model_evaluations)
    stats = calc_descriptives(gpce_regression)
    print("--------------------------------------------------------------------")
    print(stats['mean'])
    print(mean_true)
    print("--------------------------------------------------------------------")
    print(stats['variance'])
    print(variance_true)
    Spc = stats['sens_m']
    Stpc = stats['sens_t']
    print("--------------------------------------------------------------------")
    print(Spc)
    print(s_true)
    print("--------------------------------------------------------------------")
    print(Stpc)
    print(s_t_true)
    
    print("--------------------------------------------------------------------")
    S2_dict, _ = calc_sensitivity_indices(gpce_regression, 2)
    print(S2_dict)

if __name__ == "__main__":
    print(9)
    test_high_dims(dims=9)
    print(11)
    test_high_dims(dims=11)