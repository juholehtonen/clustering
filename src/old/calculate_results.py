# coding=UTF8
##################################################################
# Compare and calculate clustering results
#
# Usage: python calculate_results.py <args>
##################################################################
import pandas as pd

# Tuodaan ensin nykyinen "oikea" jako
model_dir = '../models/'
truth_file = model_dir + 'groundtruth_labels_final.csv'

labels_true = pd.read_csv(truth_file, index_col=0)


# Tuodaan laskettu kolmen klusteri

# Tarvitaan objektien yksilöivät tunnisteet!
# Lasketaan herkkyys (recall/sensitivity)
