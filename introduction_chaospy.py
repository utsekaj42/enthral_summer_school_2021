import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# === The useful help function ===
# show help for uniform distributions
#cp.Uniform?

# show help for sample generation
#cp.samplegen?
# end help

# === Distributions ===
# simple distributions
rv1 = cp.Uniform(0, 1)
rv2 = cp.Normal(0, 1)
rv3 = cp.LogNormal(0, 1, 0.2, 0.8)
print(rv1, rv2, rv3)
# end simple distributions

# joint distributions
joint_distribution = cp.J(rv1, rv2, rv3)
print(joint_distribution)
# end joint distributions

# creating iid variables
X = cp.Normal()
Y = cp.Iid(X, 4)
print(Y)
# end creating iid variables

# === Sampling ===
# sampling in chaospy
u = cp.Uniform(0,1)
#u.sample?
# end sampling chaospy

# example sampling
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)
number_of_samples = 350
samples_random = joint_distribution.sample(size=number_of_samples, rule='R')
samples_hammersley = joint_distribution.sample(size=number_of_samples, rule='M')

fig1, ax1 = plt.subplots()
ax1.set_title('Random')
ax1.scatter(*samples_random)
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')

fig2, ax2 = plt.subplots()
ax2.set_title('Hammersley sampling')
ax2.scatter(*samples_hammersley)
ax2.set_xlabel("Uniform 1")
ax2.set_ylabel("Uniform 2")
ax2.axis('equal')
# end example sampling

# example save samples to file
# Creates a csv file where each row corresponds to the sample number and each column with teh variables in the joint distribution
csv_file = "csv_samples.csv"
sep = '\t'
header = ["u1", "u2"]
header = sep.join(header)
np.savetxt(csv_file, samples_random, delimiter=sep, header=header)
# end example save samples to file

# generate external data
ext_data = np.array([sample[0] + sample[1] + sample[0]*sample[1] for sample in samples_random.T])
header = ['y0']
header = sep.join(header)
filepath = "external_evaluations.csv"
np.savetxt(filepath, ext_data, delimiter=sep, header=header)
# end generate external data

# example load samples from file
# loads a csv file where the samples/or model evaluations for each sample are saved
# with one sample per row. Multiple components ofoutput can be stored as separate columns 
filepath = "external_evaluations.csv"
data = np.loadtxt(filepath)
# end example load samples from file

# === quadrature ===
# quadrature in polychaos
#cp.generate_quadrature?
# end quadrature in polychaos

# example quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 5

nodes_gaussian, weights_gaussian = cp.generate_quadrature(order, joint_distribution, rule='G')
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order, joint_distribution, rule='C')

print('Number of nodes gaussian quadrature: {}'.format(len(nodes_gaussian[0])))
print('Number of nodes clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[1])))


fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_gaussian, marker='o', color='b')
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example quadrature

# example sparse grid quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 2
# sparse grid has exponential growth, thus a smaller order results in more points
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order, joint_distribution, rule='C')
nodes_clenshaw_sparse, weights_clenshaw_sparse = cp.generate_quadrature(order, joint_distribution, rule='C', sparse=True)

print('Number of nodes normal clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[0])))
print('Number of nodes clenshaw-curtis quadrature with sparse grid : {}'.format(len(nodes_clenshaw_sparse[0])))

fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.scatter(*nodes_clenshaw_sparse, marker= 'o', color='b')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example sparse grid quadrature

# example orthogonalization schemes
# a normal random variable
n = cp.Normal(0, 1)

x = np.linspace(0,1, 50)
# the polynomial order of the orthogonal polynomials
polynomial_order = 3

poly = cp.generate_expansion(polynomial_order, n, rule='cholesky', normed=True)
print('Cholesky decomposition {}'.format(poly))
ax = plt.subplot(131)
ax.set_title('Cholesky decomposition')
_=plt.plot(x, poly(x).T)
_=plt.xticks([])

poly = cp.generate_expansion(polynomial_order, n, rule='ttr', normed=True)
print('Discretized Stieltjes / Three terms reccursion {}'.format(poly))
ax = plt.subplot(132)
ax.set_title('Discretized Stieltjes ')
_=plt.plot(x, poly(x).T)

# TODO: this is broken
#poly = cp.generate_expansion(polynomial_order, n, rule='gram_schmidt', normed=True)
#print('Modified Gram-Schmidt {}'.format(poly))
#ax = plt.subplot(133)
#ax.set_title('Modified Gram-Schmidt')
#_=plt.plot(x, poly(x).T)
# end example orthogonalization schemes

# _Linear Regression_
# linear regression in chaospy
#cp.fit_regression?
# end linear regression in chaospy


# example linear regression
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.generate_expansion(polynomial_order, joint_distribution)

# 3.1 generate samples
number_of_samples = len(poly) #cp.bertran.terms(polynomial_order, len(joint_distribution))
samples = joint_distribution.sample(size=number_of_samples, rule='R')

# 3.2 evaluate the simple model for all samples
model_evaluations = samples[0]+samples[1]*samples[0]

# 3.3 use regression to generate the polynomial chaos expansion
gpce_regression = cp.fit_regression(poly, samples, model_evaluations)
print("Success")
# end example linear regression


# _Spectral Projection_
# spectral projection in chaospy
# cp.fit_quadrature?
# end spectral projection in chaospy


# example spectral projection
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.generate_expansion(polynomial_order, joint_distribution)

# 4.1 generate quadrature nodes and weights
order = 5
nodes, weights = cp.generate_quadrature(order, joint_distribution, rule='G')

# 4.2 evaluate the simple model for all nodes
model_evaluations = nodes[0]+nodes[1]*nodes[0]

# 4.3 use quadrature to generate the polynomial chaos expansion
gpce_quadrature = cp.fit_quadrature(poly, nodes, weights, model_evaluations)
print("Success")
# end example spectral projection

# example uq
# TODO: note chaospy's implementation of E, Std, Sens_m and Sens_t is very
# inefficient for high dimensional/high order approximations.
# chaospy.wrapper.py provides a fast wrapper to calculate these from the
# regression/quadrature END TODO
exp_reg = cp.E(gpce_regression, joint_distribution)
exp_ps =  cp.E(gpce_quadrature, joint_distribution)

std_reg = cp.Std(gpce_regression, joint_distribution)
str_ps = cp.Std(gpce_quadrature, joint_distribution)

prediction_interval_reg = cp.Perc(gpce_regression, [5, 95], joint_distribution)
prediction_interval_ps = cp.Perc(gpce_quadrature, [5, 95], joint_distribution)

print("Expected values   Standard deviation            90 % Prediction intervals\n")
print(' E_reg |  E_ps     std_reg |  std_ps                pred_reg |  pred_ps')
print('  {} | {}       {:>6.3f} | {:>6.3f}       {} | {}'.format(exp_reg,
                                                                  exp_ps,
                                                                  std_reg,
                                                                  str_ps,
                                                                  ["{:.3f}".format(p) for p in prediction_interval_reg],
                                                                  ["{:.3f}".format(p) for p in prediction_interval_ps]))
# end example uq

# example sens
sensFirst_reg = cp.Sens_m(gpce_regression, joint_distribution)
sensFirst_ps = cp.Sens_m(gpce_quadrature, joint_distribution)

sensT_reg = cp.Sens_t(gpce_regression, joint_distribution)
sensT_ps = cp.Sens_t(gpce_quadrature, joint_distribution)

print("First Order Indices           Total Sensitivity Indices\n")
print('       S_reg |  S_ps                 ST_reg |  ST_ps  \n')
for k, (s_reg, s_ps, st_reg, st_ps) in enumerate(zip(sensFirst_reg, sensFirst_ps, sensT_reg, sensT_ps)):
    print('S_{} : {:>6.3f} | {:>6.3f}         ST_{} : {:>6.3f} | {:>6.3f}'.format(k, s_reg, s_ps, k, st_reg, st_ps))
# end example sens

# example exact solution
import sympy as sp
import sympy.stats
from sympy.utilities.lambdify import lambdify, implemented_function

pdf_beta = lambda b: 1
support_beta = (pdf_beta,0,1)
         
pdf_chi = lambda x:  1
support_chi = (pdf_chi,0, 1)
x, b = sp.symbols("x, b")
y = x + x*b

support_beta = (b,0,1)
support_chi = (x,0,1)
mean_g_beta = sp.Integral(y*pdf_chi(x), support_chi)
mean_g_chi =  sp.Integral(y*pdf_beta(b), support_beta)
mean = sp.Integral(mean_g_beta*pdf_beta(b), support_beta)
print("Expected value {}".format(mean.doit()))
variance = sp.Integral(pdf_beta(b)*sp.Integral(pdf_chi(x)*(y-mean)**2,support_chi), support_beta)
print("Variance: {}".format(variance.doit()))
var_E_g_beta = sp.Integral(pdf_beta(b)*(mean_g_beta-mean)**2, support_beta)
var_E_g_chi = sp.Integral(pdf_chi(x)*(mean_g_chi-mean)**2, support_chi)

S_chi =  var_E_g_chi/variance
S_beta = var_E_g_beta/variance


print("S_beta {}".format(S_beta.doit()))
print("S_chi {}".format(S_chi.doit()))
# end example exact solution
