import matplotlib.cm as cm
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics


class DisciplineExtractor(BaseEstimator, TransformerMixin):
    """Extract the discipline from data"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(d['discipline'])
        return transformed


class GeneralExtractor(BaseEstimator, TransformerMixin):
    """Extract and merge all fields from a sample to a string
    Results:
    [' The decrease of visual acuity in cataract patients waiting for
     surgery Purpose: To investigate the rapidity of vision loss in
     eyes waiting for cataract surgery and to estimate what proportion
     of life expectancy the extended wait for surgery comprised.Methods:
     The visual acuities at the time of referral and on the day
     beforesurgery were compared in 124 patients operated on for
     cataract in the Vaasa Central Hospital, Finland. The expected
     survival of the patients after surgery was calculated individually
     using the Finnish life statistics.Results: During the waiting time
     of 13 months on the average, the visual acuity in the study eye
     decreased from 0.68 logMAR (0.2 in decimal values) to 0.96 logMAR
     (0.1). The average decrease in vision was 0.27 logMAR per year
     varying from none to 2.07 logMAR units. 30% of the eyes experienced
     worsening of vision by 60% or more while 48% had no or minimal
     worsening (<0.2 logMAR). The rapidity of change in visual acuity
     was somewhat less in older patients (75 years or older), but the
     difference was not statistically significant. The percentage of
     persons with visual acuity of 0.5 or better in the better eye
     decreased from 66% to 41% and those with low vision (<0.3 in the
     better eye) increased from 8% to 21%. The mean waiting time in
     relationto the expected survival for all patients was 13% varying
     from less than 5% in 10 patients to more than 25% in 8
     patients.Conclusion: Progression of vision loss in eyes waiting
     for cataract surgery varies significantly. For many patients the
     extended delay caused remarkable disability for a considerable
     part of their remaining lifetime.\n cataract_surgery waiting_time
     visual_acuity logMAR_values progression_of_cataract life_expectancy',
     '...',...]


    :param data: list of dicts
    :param fields: list of fields to be analyzed
    """
    def __init__(self, fields=None):
        self.fields = fields

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            if self.fields:
                bag_of_words = ''
                for field in self.fields:
                    bag_of_words += ' ' + d.get(field, '')
                transformed.append(bag_of_words)
            else:
                raise IndexError('No items in "fields".')
        return transformed
            #yield bag_of_words


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



def identity_tokenizer(doc):
    return doc

def plot_silhouette(X, labels, n_clusters, method):
    if method == 'K-Means':
        try:
            X = X.todense()
        except Exception as e:
            return False

    fig, ax1 = plt.subplots()
    fig.set_size_inches(16, 14)

    # Set the silhouette coefficient range
    min_x = -0.15
    ax1.set_xlim([min_x, 1])
    # Insert some blank space with (n_clusters+1)*10 to demarcate clusters
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(X, labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(min_x - 0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Mark singleton/thin clusters
        if size_cluster_i in [1, 2, 3]:
            ax1.text(ax1.get_xlim()[1] - 0.05, y_lower + 0.5 * size_cluster_i, str(size_cluster_i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # xticks = np.linspace(min_x, 1, ((1 - min_x)/0.2) + 1 )
    xticks = [-0.15, 0, 0.2, 0.4, 0.6, 0.8, 1]
    ax1.set_xticks(xticks)

    plt.suptitle(("Silhouette analysis for {0} clustering with n_clusters = {1}"
                  .format(method, n_clusters)), fontsize=14, fontweight='bold')
    n_comp = X.shape[1] if X.shape[1] < 600 else '-'
    plt.savefig('../img/{0}-{1}-{2}-{3}-silhouette-plot.png'.format(len(labels), n_clusters, n_comp, method), bbox_inches='tight')
    #plt.show()