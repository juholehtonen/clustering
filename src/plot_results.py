# encoding: utf-8
##################################################################
# Scrap silhouette values from logs and plot
#
# Run this script in the processed/results folder.'
# $ python ../../src/plot_results.py
#
# Produces three files:
# silhouette-coefficients-12000-2_260-800-kmeans.txt
# calinski-harabasz-12000-2_260-800-kmeans.txt
# c-h-silh-index-plot-12000-2_260-800-kmeans.png
#
# Original grep expressions
# grep -i 'silhouette coefficient: ' 12000-*{0}*| sort -V | sed -e 's/[0-9]*-\([0-9]*\)-.*cient: \(.*\)/\1 \2/' > silhouette-coefficients-{1}.txt".format(method, params))
# grep -i 'Calinski-Harabasz Index: ' 12000-*{0}*| sort -V | sed -e 's/[0-9]*-\([0-9]*\)-.*Index: \(.*\)/\1 \2/' > calinski-harabasz-{1}.txt".format(method, params))
##################################################################
from distutils.version import LooseVersion
import numpy as np
import os
import pylab as pl
import re


size = 12000
k_min = 2
k_max = 260
n_compnents = 800
# method = 'hierarchical'
method = 'kmeans'
params = '{0}-{1}_{2}-{3}-{4}'.format(size, k_min, k_max, n_compnents, method)
# results_dir = '../data/baseline/results/'
results_dir = './'
silh_file = results_dir + 'silhouette-coefficients-{0}.txt'.format(params)
cali_file = results_dir + 'calinski-harabasz-{0}.txt'.format(params)
plt_sym = '.'

file_list = os.listdir(results_dir)
pattern_method = r'12000-.*{0}.*txt'.format(method)
files_filtered = [f for f in file_list if re.match(pattern_method, f)]
files_sorted = sorted(files_filtered, key=LooseVersion)

pattern_k = r'[0-9]*-([0-9]*)-[0-9]*.*'
pattern_silh = r'.* Silhouette Coefficient: (.*)'
pattern_ch = r'.* Calinski-Harabasz Index: (.*)'
silh_vals = []
c_h_vals = []
for fs in files_sorted:
    match_k = re.match(pattern_k, fs)
    if match_k:
        with open(fs) as fo:
            for line in fo:
                match_silh = re.match(pattern_silh, line)
                if match_silh:
                    silh_vals.append([int(match_k[1]), float(match_silh[1])])
                match_ch = re.match(pattern_ch, line)
                if match_ch:
                    c_h_vals.append([int(match_k[1]), float(match_ch[1])])
    else:
        raise ValueError
silh_arr = np.array(silh_vals)
c_h_arr = np.array(c_h_vals)

with open(silh_file, 'w') as handle:
    np.savetxt(handle, silh_vals)
with open(cali_file, 'w') as handle:
    np.savetxt(handle, c_h_arr)

# Plot 'Silhouette value'
# pl.plot(silh_arr[:,0], silh_arr[:,1],'r.')
# pl.xlabel('N of clusters')
# pl.ylabel('Silhouette value')
# pl.ylim(0, max(silh_vals)*2)
# pl.show()
# pl.clf()

# Plot both indices
f, axarr = pl.plt.subplots(2, sharex=True)
axarr[0].plot(c_h_arr[:,0], c_h_arr[:,1], 'r' + plt_sym)
axarr[0].set_title('Calinski-Harabasz index')
# axarr[0].set_ylim(min(c_h_vals)-2, max(c_h_vals)+2)
axarr[1].set_title('Silhouette value')
axarr[1].plot(silh_arr[:,0], silh_arr[:,1], 'r' + plt_sym)
# axarr[1].set_ylim(0, 0.1)
axarr[1].set_xlabel('N of clusters')

f.savefig(results_dir + 'c-h-silh-index-plot-{0}.png'.format(params))