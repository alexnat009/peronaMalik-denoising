import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from numericalODE import *

f = lambda t, s: np.exp(-t)
h = 0.1
t = np.arange(0, 2 + h, h)
s0 = -1

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), layout='tight')  # type:figure.Figure, axes.Axes
t1, s1 = runge_kutta4(f, t, h, s0)
t2, s2 = fourth_order_adams_bashforth(f, t, h, s0)
t3, s3 = fourth_order_adams_moulton(f, t, h, s0)
t4, s4 = three_stage_diagonally_implicit_runge_kutta_method(f, t, h, s0)

ax[0][0].plot(t1, s1, 'bo--', label='Approximate')
ax[0][0].plot(t1, -np.exp(-t1), 'g', label='Exact')
ax[0][0].set_title('Runge Kutta 4')

ax[0][1].plot(t2, s2, 'ro--', label='Approximate')
ax[0][1].plot(t2, -np.exp(-t2), 'g', label='Exact')
ax[0][1].set_title('Adams Bashforth')

ax[1][0].plot(t3, s3, 'go--', label='Approximate')
ax[1][0].plot(t3, -np.exp(-t3), 'g', label='Exact')
ax[1][0].set_title('Adams Moulton')

ax[1][1].plot(t4, s4, 'yo--', label='Approximate')
ax[1][1].plot(t4, -np.exp(-t4), 'g', label='Exact')
ax[1][1].set_title('Three Stage Diagonally Implicit Runge Kutta')

x_limits = [-1, 2]
y_limits = [-1.5, 1]
for axi in ax.flat:
    axi.set_xlim(x_limits)
    axi.set_ylim(y_limits)
    axi.legend()

plt.tight_layout()
plt.show()
