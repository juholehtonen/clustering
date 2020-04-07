# encoding: utf-8
##################################################################
# Scrap silhouette values from logs and plot
#
#
##################################################################
import numpy as np
import os
import pylab as pl


# grep -i "silhouette coefficient: " 12000-*| cut -f7 -d ' ' > silhouette-coefficients-12000-2_14-800-kmeans.txt
# grep -i "Calinski-Harabasz Index: " 12000-*| cut -f7 -d ' ' -s > calinski-harabasz-12000-2_14-800-kmeans.txt

x_range = range(2, 16)
params = '{0}-{1}_{2}-{3}'.format(519, 2, 14, 800)
results = '../data/baseline/results/'
silh = results + 'silhouette-coefficients-12000-2_14-800-kmeans.txt'
cali = results + 'calinski-harabasz-12000-2_14-800-kmeans.txt'

silh_vals = np.loadtxt(silh)
c_h_vals = np.loadtxt(cali)

# Plot 'Silhouette value'
pl.plot(x_range, silh_vals,'r-')
pl.xlabel('N of clusters')
pl.ylabel('Silhouette value')
pl.ylim(0, max(silh_vals)*2)
pl.show()
# pl.clf()

# Plot both indices
f, axarr = pl.plt.subplots(2, sharex=True)
axarr[0].plot(x_range, c_h_vals,'r-')
axarr[0].set_title('Calinski-Harabasz index')
axarr[0].set_ylim(min(c_h_vals)-2, max(c_h_vals)+2)
axarr[1].set_title('Silhouette value')
axarr[1].plot(x_range, silh_vals,'r-')
axarr[1].set_ylim(0, 0.1)
axarr[1].set_xlabel('N of clusters')

f.savefig(results + 'c-h-silh-index-plot-{0}.png'.format(params))