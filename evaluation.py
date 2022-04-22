from __future__ import division
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, normalized_mutual_info_score
import numpy as np

def clustering_latent_space(emb, label, nb_clusters=None):
    """ Node Clustering: computes Adjusted Mutual Information score from a
    K-Means clustering of nodes in latent embedding space
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :param label: ground-truth node labels
    :param nb_clusters: int number of ground-truth communities in graph
    :return: Adjusted Mutual Information (AMI) score
    """
    if nb_clusters is None:
        nb_clusters = len(np.unique(label))
    # K-Means Clustering
    km = KMeans(n_clusters = nb_clusters,max_iter=500)
    clustering_pred = km.fit_predict(emb)
    # Compute metrics
    return clustering_pred, km.cluster_centers_
