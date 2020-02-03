# encoding: utf-8
##################################################################
# Scrap silhouette values from logs and plot
#
#
##################################################################
import numpy as np
import os
import pylab as pl


processed = os.listdir('../data/processed/')
silhouette_file = 'silhouette-coefficients-12000-200_259-800-kmeans.txt'
c_h_file = 'calinski-harabasz-12000-200_259-800-kmeans.txt'

silh_vals = np.loadtxt(silhouette_file)
c_h_vals = np.loadtxt(c_h_file)

pl.plot(range(200,260), silh_vals,'r-')
pl.xlabel('N of clusters')
pl.ylabel('Silhouette value')
pl.ylim(0, max(silh_vals)*2)
pl.show()
# pl.clf()


f, axarr = pl.plt.subplots(2, sharex=True)
axarr[0].plot(range(200,260), c_h_vals,'r-')
axarr[0].set_title('Calinski-Harabasz index')
axarr[0].set_ylim(10, 20)
axarr[1].set_title('Silhouette value')
axarr[1].plot(range(200,260), silh_vals,'r-')
axarr[1].set_ylim(0, 0.1)
axarr[1].set_xlabel('N of clusters')

f.savefig('c-h-silh-index-plot-12000-200_259-800.png')