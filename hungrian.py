'''
If we use (external) classification evalutation measures like F1 or
accuracy for clustering evaluation, problems may arise.

One way to fix is to perform label matching.

Here we performs kmeans clustering on the Iris dataset and proceed to use
the Hungarian (Munkres) algorithm to correct the mismatched labeling.

https://gist.github.com/siolag161/dc6e42b64e1bde1f263b
'''

import sys
import numpy as np
from munkres import Munkres

def make_cost_matrix(c1, c2):

    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    # print('uc1,uc2:',uc1,uc2)
    # print('l1,l2:',l1,l2)
    assert(l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m

def translate_clustering(clt, mapper):
    return np.array([ mapper[i] for i in clt ])

def label_mapping(tru,pre):

    classes = tru
    labels = pre
    num_labels = len(np.unique(classes))
    cost_matrix = make_cost_matrix(labels, classes)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = { old: new for (old, new) in indexes }
    new_labels = translate_clustering(labels, mapper)

    return new_labels


