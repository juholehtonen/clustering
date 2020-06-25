# encoding: utf-8
##################################################################
# Scrap silhouette values from logs and plot
#
# Run this script in the processed/results folder.'
# Remeber to change 'method'.
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


# size = 12000
size = 455
k_min = 2
k_max = 12
n_components = 800
method = 'hierarchical'
# method = 'kmeans'
params = '{0}-{1}_{2}-{3}-{4}'.format(size, k_min, k_max, n_components, method)
# results_dir = '../data/baseline/results/'
results_dir = './'

# Directory for plots and validation indices
image_dir = '../images/'
silh_file = image_dir + 'silhouette-coefficients-{0}.txt'.format(params)
cali_file = image_dir + 'calinski-harabasz-{0}.txt'.format(params)
ari_file = image_dir + 'adjusted_rand-index-{0}.txt'.format(params)
sdbw_file = image_dir + 's_dbw_validity-index-{0}.txt'.format(params)
# plt_sym = '.'
plt_sym = '-'

file_list = os.listdir(results_dir)
pattern_method = r'12000-.*{0}.*txt'.format(method)
files_filtered = [f for f in file_list if re.match(pattern_method, f)]
files_sorted = sorted(files_filtered, key=LooseVersion)

pattern_k = r'[0-9]*-([0-9]*)-[0-9]*.*'
pattern_silh = r'.* Silhouette Coefficient: (.*)'
pattern_ch = r'.* Calinski-Harabasz Index: (.*)'
pattern_ari = r'.* Adjusted Rand-Index: (.*)'
pattern_sdbw = r'.* S_Dbw validity index: (.*)'
silh_vals = []
c_h_vals = []
ari_vals = []
sdbw_vals = []
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
                match_ari = re.match(pattern_ari, line)
                if match_ari:
                    ari_vals.append([int(match_k[1]), float(match_ari[1])])
                match_sdbw = re.match(pattern_sdbw, line)
                if match_sdbw:
                    sdbw_vals.append([int(match_k[1]), float(match_sdbw[1])])
    else:
        raise ValueError
silh_arr = np.array(silh_vals)
c_h_arr = np.array(c_h_vals)
ari_arr = np.array(ari_vals)
sdbw_arr = np.array(sdbw_vals)

with open(silh_file, 'w') as handle:
    np.savetxt(handle, silh_vals)
with open(cali_file, 'w') as handle:
    np.savetxt(handle, c_h_arr)
with open(ari_file, 'w') as handle:
    np.savetxt(handle, ari_arr)
with open(sdbw_file, 'w') as handle:
    np.savetxt(handle, sdbw_arr)

# Plot 'Silhouette value'
# pl.plot(silh_arr[:,0], silh_arr[:,1],'r.')
# pl.xlabel('N of clusters')
# pl.ylabel('Silhouette value')
# pl.ylim(0, max(silh_vals)*2)
# pl.show()
# pl.clf()

# Plot indices
xticks = np.arange(k_min, k_max+1)
f, ((ax1, ax2), (ax3, ax4)) = pl.plt.subplots(2, 2, sharex=True)
ax1.plot(c_h_arr[:k_max, 0], c_h_arr[:k_max, 1], 'r' + plt_sym)
ax1.set_title('Calinski-Harabasz index')
ax1.set_xticks(xticks)
ax2.plot(silh_arr[:k_max, 0], silh_arr[:k_max, 1], 'r' + plt_sym)
ax2.set_title('Silhouette value')
ax3.plot(sdbw_arr[:k_max, 0], sdbw_arr[:k_max, 1], 'r' + plt_sym)
ax3.set_title('S_Dbw validity index')
ax4.plot(ari_arr[:k_max, 0], ari_arr[:k_max, 1], 'r' + plt_sym)
ax4.set_title('Adjusted Rand-index')
ax3.set_xlabel('Number of clusters')
ax4.set_xlabel('Number of clusters')

f.savefig(image_dir + 'c-h-silh-index-plot-{0}.png'.format(params))