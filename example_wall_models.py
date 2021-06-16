import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import chaospy_wrapper as cpw
from monte_carlo import generate_sample_matrices_mc
from monte_carlo import calculate_sensitivity_indices_mc
unit_mmhg_pa = 133.3
unit_pa_mmhg = 1./unit_mmhg_pa

unit_cm2_m2 = 1. / 100. / 100.
unit_m2_cm2 = 1. / unit_cm2_m2

# begin quadratic area model
def quadratic_area_model(pressure_range, samples):
    """
    calculate the arterial vessel wall for a set pressure range
    from 75 to 140 mmHg for a given set of reference wave speeds
    area and pressure.

    :param
        pressure_range: np.array
            pressure range for which to calculate the arterial area

        samples: np.array with shape (4:n_samples)
            sample matrix where
            first indices correspond to
                a_s: reference area
                c_s: reference waves seed
                p_s: reference pressure
                rho: blood density
            and n_samples to the number of samples

    :return:
        arterial pressure : np.array
            of size n_samples
    """
    pressure_range = pressure_range.reshape(pressure_range.shape[0], 1)
    a_s, c_s, p_s, rho = samples
    beta = 2*rho*c_s**2/np.sqrt(a_s)*a_s
    #C_Laplace = (2. * ((P - Ps) * As / betaLaplace + np.sqrt(As))) * As / betaLaplace
    return ((pressure_range - p_s)*a_s/beta + np.sqrt(a_s))**2.
# end quadratic area model

# begin exponential area model
def logarithmic_area_model(pressure_range, samples):
    """
    calculate the arterial vessel wall for a set pressure range
    from 75 to 140 mmHg for a given set of reference wave speeds
    area and pressure.

    :param
        pressure_range: np.array
            pressure range for which to calculate the arterial area

        samples: np.array with shape (4:n_samples)
            sample matrix where
            first indices correspond to
                a_s: reference area
                c_s: reference waves seed
                p_s: reference pressure
                rho: blood density
            and n_samples to the number of samples

    :return:
        arterial pressure : np.array
            of size n_samples
    """
    pressure_range = pressure_range.reshape(pressure_range.shape[0], 1)
    a_s, c_s, p_s, rho = samples
    beta = 2.0*rho*c_s**2./p_s
    #C_hayashi = 2.0 * As / betaHayashi * (1.0 + np.log(P / Ps) / betaHayashi) / P
    return a_s*(1.0 + np.log(pressure_range / p_s)/beta) ** 2.0
# end exponential area model

# start deterministic comparison
pressure_range = np.linspace(45, 180, 100) * unit_mmhg_pa
a_s = 5.12 * unit_cm2_m2
c_s = 6.25609258389
p_s = 100 * unit_mmhg_pa
rho = 1045.

plt.figure()
for model, name in [(quadratic_area_model, 'Quadratic model'), (logarithmic_area_model, 'Logarithmic model')]:
    y_area = model(pressure_range, (a_s, c_s, p_s, rho))
    plt.plot(pressure_range * unit_pa_mmhg, y_area * unit_m2_cm2, label=name)
    plt.xlabel('Pressure [mmHg]')
    plt.ylabel('Area [cm2]')
    plt.legend()
plt.tight_layout()
# end deterministic comparison

# Create marginal and joint distributions
dev = 0.05
a_s = 5.12 * unit_cm2_m2
A_s = cp.Uniform(a_s * (1. - dev), a_s*(1. + dev))

c_s = 6.25609258389
C_s = cp.Uniform(c_s * (1. - dev), c_s*(1. + dev))

p_s = 100 * unit_mmhg_pa
P_s = cp.Uniform(p_s * (1. - dev), p_s*(1. + dev))

rho = 1045.
Rho = cp.Uniform(rho * (1. - dev), rho*(1. + dev))

jpdf = cp.J(A_s, C_s, P_s, Rho)
# End Create marginal and joint distributions


# scatter plots
pressure_range = np.linspace(45, 180, 100) * unit_mmhg_pa
Ns = 200
sample_scheme = 'R'
samples = jpdf.sample(Ns, sample_scheme)

for model, name, color in [(quadratic_area_model, 'Quadratic model', '#dd3c30'), (logarithmic_area_model, 'Logarithmic model', '#2775b5')]:
    # evaluate the model for all samples
    Y_area = model(pressure_range, samples)

    plt.figure()
    plt.title('{}: evaluations Y_area'.format(name))
    plt.plot(pressure_range * unit_pa_mmhg, Y_area * unit_m2_cm2)
    plt.xlabel('Pressure [mmHg]')
    plt.ylabel('Area [cm2]')
    plt.ylim([3.5,7.])

    plt.figure()
    plt.title('{}: scatter plots'.format(name))
    for k in range(len(jpdf)):
        plt.subplot(len(jpdf)/2, len(jpdf)/2, k+1)
        plt.plot(samples[k], Y_area[0]*unit_m2_cm2, '.', color=color)
        plt.ylabel('Area [cm2]')
        plt.ylim([3.45, 4.58])
        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)
    plt.tight_layout()
# end scatter plots


# start Monte Carlo
pressure_range = np.linspace(45, 180, 100) * unit_mmhg_pa
Ns = 50000
sample_method = 'R'
number_of_parameters = len(jpdf)

# 1. Generate sample matrices
A, B, C = generate_sample_matrices_mc(Ns, number_of_parameters, jpdf, sample_method)

fig_mean = plt.figure('mean')
ax_mean = fig_mean.add_subplot(1,2,1)
ax_dict = {}
figs = {}
for name in ['Quadratic model', 'Logarithmic model'] :
    figs[name] = plt.figure(name)
    ax = ax_dict[name]= figs[name].add_subplot(1,2,1)
    
print("\n Uncertainty measures (averaged)\n")
print('\n  E(Y)  |  Std(Y) \n')
for model, name, color in [(quadratic_area_model, 'Quadratic model', '#dd3c30'),
                           (logarithmic_area_model, 'Logarithmic model', '#2775b5')]:
    # evaluate the model for all samples
    Y_A = model(pressure_range, A.T)
    Y_B = model(pressure_range, B.T)

    Y_C = np.empty((len(pressure_range), Ns, number_of_parameters))
    for i in range(number_of_parameters):
        Y_C[:, :, i] = model(pressure_range, C[i, :, :].T)

    # calculate statistics
    plotMeanConfidenceAlpha = 5.
    Y = np.hstack([Y_A, Y_B])
    expected_value = np.mean(Y, axis=1)
    variance = np.var(Y, axis=1)
    std = np.sqrt(np.var(Y, axis=1))
    prediction_interval = np.percentile(Y, [plotMeanConfidenceAlpha / 2., 100. - plotMeanConfidenceAlpha / 2.], axis=1)

    print('{:2.5f} | {:2.5f} : {}'.format(np.mean(expected_value)* unit_m2_cm2, np.mean(std)* unit_m2_cm2, name))

    # 3. Approximate the sensitivity indices
    # Better to centralize data before calculating sensitivities
    expected_value_col = expected_value.copy()
    expected_value_col.shape = (expected_value.shape[0],1)
    Y_Ac = Y_A - expected_value_col
    Y_Bc = Y_B  - expected_value_col
    Y_Cc = Y_C.transpose() - expected_value
    Y_Cc = Y_Cc.transpose()
    S, ST = calculate_sensitivity_indices_mc(Y_A, Y_B, Y_C)
    Sc, STc = calculate_sensitivity_indices_mc(Y_Ac, Y_Bc, Y_Cc)

    ## Plots
    fig = fig_mean
    ax_mean = ax_mean
    ax_mean.plot(pressure_range * unit_pa_mmhg, expected_value * unit_m2_cm2, label=name, color=color)
    ax_mean.fill_between(pressure_range * unit_pa_mmhg, prediction_interval[0] * unit_m2_cm2,
                     prediction_interval[1] * unit_m2_cm2, alpha=0.3, color=color)
    ax_mean.set_xlabel('Pressure [mmHg]')
    ax_mean.set_ylabel('Area [cm2]')
    ax_mean.legend()
    fig.tight_layout()

    fig = figs[name]
    ax = ax_dict[name]
    ax.set_title('{}: sensitvity indices'.format(name))
    colorsPie = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    labels = ['Area', 'wave speed', 'pressure', 'density']
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    #firstOrderIndicesTimeAverage = S[:, 50]
    #totalIndicesTimeAverage = ST[:, 50]
    #rects1 = plt.bar(ind, firstOrderIndicesTimeAverage, width, color=colorsPie, alpha=0.5)
    #rects2 = plt.bar(ind + width, totalIndicesTimeAverage, width, color=colorsPie, hatch='.')

    firstOrderIndicesTimeAverage = np.mean(S*variance,axis=1)/np.mean(variance)
    totalIndicesTimeAverage = np.mean(ST*variance,axis=1)/np.mean(variance)
    rects1 = ax.bar(ind, firstOrderIndicesTimeAverage, width, color=colorsPie, alpha = 0.5)
    rects2 = ax.bar(ind + width, totalIndicesTimeAverage, width, color=colorsPie, hatch = '.')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Si')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)
    ax.set_xlim([-width, N + width])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', top='off')
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', right='off')

    fig.tight_layout()
# end Monte Carlo

# start polynomial chaos
# orthogonal C_polynomial from marginals
polynomial_order = 3
#Ns = 2*cp.bertran.terms(polynomial_order, len(jpdf))

print("\n Uncertainty measures (averaged)\n")
print('\n  E(Y)  |  Std(Y) \n')



ax_mean = fig_mean.add_subplot(1,2,2)
ax_dict = {}
for name in ['Quadratic model', 'Logarithmic model'] :
    figs[name] = plt.figure(name)
    ax = ax_dict[name]= figs[name].add_subplot(1,2,2)
    
for model, name, color in [(quadratic_area_model, 'Quadratic model', '#dd3c30'),
                           (logarithmic_area_model, 'Logarithmic model', '#2775b5')]:
    sample_scheme = 'R'
    # create orthogonal polynomials
    basis = cpw.generate_basis(polynomial_order, jpdf)
    # create samples
    Ns = 2*len(basis['poly'])
    samples = jpdf.sample(Ns, sample_scheme)
    # evaluate the model for all samples
    Y_area = model(pressure_range, samples)
    # polynomial chaos expansion
    polynomial_expansion = cpw.fit_regression(basis, samples, Y_area.T)

    # calculate statistics
    plotMeanConfidenceAlpha = 5
    expected_value = cpw.E(polynomial_expansion, jpdf)
    variance = cpw.Var(polynomial_expansion, jpdf)
    standard_deviation = cpw.Std(polynomial_expansion, jpdf)
    prediction_interval = cpw.Perc(polynomial_expansion,
                                  [plotMeanConfidenceAlpha / 2., 100 - plotMeanConfidenceAlpha / 2.],
                                  jpdf)
    print('{:2.5f} | {:2.5f} : {}'.format(np.mean(expected_value) * unit_m2_cm2, np.mean(std) * unit_m2_cm2, name))

    # compute sensitivity indices
    S = cpw.Sens_m(polynomial_expansion, jpdf)
    ST = cpw.Sens_t(polynomial_expansion, jpdf)

    fig = fig_mean
    ax_mean =  ax_mean
    ax_mean.plot(pressure_range * unit_pa_mmhg, expected_value * unit_m2_cm2, label=name, color=color)
    ax_mean.fill_between(pressure_range * unit_pa_mmhg, prediction_interval[0] * unit_m2_cm2,
                     prediction_interval[1] * unit_m2_cm2, alpha=0.3, color=color)
    ax_mean.set_xlabel('Pressure [mmHg]')
    ax_mean.set_ylabel('Area [cm2]')
    ax_mean.legend()
    fig.tight_layout()

    fig = figs[name]
    ax = ax_dict[name]
    ax.set_title('{}: sensitvity indices'.format(name))
    colorsPie = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    labels = ['Area', 'wave speed', 'pressure', 'density']
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    #firstOrderIndicesTimeAverage = S[:, 50]
    #totalIndicesTimeAverage = ST[:, 50]
    #rects1 = plt.bar(ind, firstOrderIndicesTimeAverage, width, color=colorsPie, alpha=0.5)
    #rects2 = plt.bar(ind + width, totalIndicesTimeAverage, width, color=colorsPie, hatch='.')

    firstOrderIndicesTimeAverage = np.mean(S*variance,axis=1)/np.mean(variance)
    totalIndicesTimeAverage = np.mean(ST*variance,axis=1)/np.mean(variance)
    rects1 = ax.bar(ind, firstOrderIndicesTimeAverage, width, color=colorsPie, alpha = 0.5)
    rects2 = ax.bar(ind + width, totalIndicesTimeAverage, width, color=colorsPie, hatch = '.')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Si')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)
    ax.set_xlim([-width, N + width])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', top='off')
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', right='off')

    fig.tight_layout()
# end polynomial chaos

plt.show()

