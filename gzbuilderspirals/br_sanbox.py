import numpy as np
# from sklearn.linear_model import BayesianRidge, ARDRegression
import matplotlib.pyplot as plt
from gzbuilderspirals import fitting, xy_from_r_theta
from gzbuilderspirals import

R, t = np.load('./gzbuilderspirals/messy-r-t.npy')
# R = np.linspace(0.01, 0.2, 500)
# t = np.log(R)*-1.4 + 4
# noise = np.random.randn(R.shape[0]) * 0.1 * 1/np.sqrt(R)
# t += noise

log_spiral_result = fitting.log_spiral_fit(R, t)
poly_spiral_result = fitting.polynomial_fit(R, t, degree=7)

plt.plot(R, t, '.', markersize=3)
plt.plot(log_spiral_result['R'], log_spiral_result['T'])
plt.fill_between(
    log_spiral_result['R'],
    log_spiral_result['T'] - log_spiral_result['T_std'],
    log_spiral_result['T'] + log_spiral_result['T_std'],
    color='C1', alpha=0.2
)
plt.plot(poly_spiral_result['R'], poly_spiral_result['T'])
plt.fill_between(
    poly_spiral_result['R'],
    poly_spiral_result['T'] - poly_spiral_result['T_std'],
    poly_spiral_result['T'] + poly_spiral_result['T_std'],
    color='C1', alpha=0.2
)
# plt.xscale('log')

# print('Optimal k:', poly_spiral_result['k'])

print('Polynomial AIC:', poly_spiral_result['AIC'])
print('Log Spiral AIC:', log_spiral_result['AIC'])
print(
    'Prefer',
    (
        'polynomial'
        if poly_spiral_result['AIC'] < log_spiral_result['AIC']
        else 'log spiral'
    )
)
plt.show()

plt.figure()
x, y = xy_from_r_theta(R, t)
log_xy = xy_from_r_theta(
    log_spiral_result['R'],
    log_spiral_result['T']
)
poly_xy = xy_from_r_theta(
    poly_spiral_result['R'],
    poly_spiral_result['T']
)
plt.plot(x, y, '.')
plt.plot(*log_xy)
plt.plot(*poly_xy)
