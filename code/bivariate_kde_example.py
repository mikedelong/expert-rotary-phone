import numpy as np
from matplotlib.pyplot import figure
from matplotlib.pyplot import show
from scipy.stats import gaussian_kde

data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
x = data[:, 0]
y = data[:, 1]

# Perform the kernel density estimate
xx, yy = np.mgrid[-3:3:100j, -3:3:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = figure()
ax = fig.gca()
# Contourf plot
cfset = ax.contourf(xx, yy, f, cmap='Blues')
## Or kernel density estimate plot instead of the contourf plot
ax.imshow(np.rot90(f), cmap='Blues',
          # extent=[xmin, xmax, ymin, ymax]
          )
# Contour plot
cset = ax.contour(xx, yy, f, colors='k')
# Label plot
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Y1')
ax.set_ylabel('Y0')

show()
