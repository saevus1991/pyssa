import numpy as np
import matplotlib.pyplot as plt
from pyssa.util import softmax_stepfun, logistic_stepfun

# create time_grid
time_grid = np.sort(100*np.random.rand(11))
vals = np.round(2*np.random.rand(10)-1.0)
vals = 5+np.cumsum(vals)

# create linear space
beta = 5.0
time = np.linspace(time_grid[0], time_grid[-1], 1000)
val_interp = np.zeros(time.shape)
for i, t in enumerate(time):
    val_interp[i] = logistic_stepfun(t, time_grid, vals, beta)


plt.step(time_grid[0:-1], vals, '-b', where='post')
plt.plot(time, val_interp, '-r')
plt.show()