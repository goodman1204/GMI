import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score
from sklearn import metrics
import itertools
import os
from collections import Counter
from munkres import Munkres, print_matrix

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy

def find_motif(adj, dataset_name):

    path = 'data/{}_motif.npy'.format(dataset_name)
    motif_matrix = None

    if os.path.exists(path):
        motif_matrix = np.load(path,allow_pickle=True)
    else:

        g = nx.Graph()
        g = nx.from_scipy_sparse_matrix(adj)
        target = nx.Graph()
        target.add_edge(1,2)
        target.add_edge(2,3)

        N = g.number_of_nodes()
        motif_matrix = np.zeros((N,N))

        for node in g.nodes():
            print(node)
            neigbours = [i for i in g.neighbors(node)]
            for sub_nodes in itertools.combinations(neigbours,len(target.nodes())):
                subg = g.subgraph(sub_nodes)
                if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                    for e in subg.edges():
                        motif_matrix[e[0]][e[1]]=1
                        motif_matrix[e[1]][e[0]]=1

        with open(path,'wb') as wp:
            np.save(wp,motif_matrix)

    return motif_matrix



def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(rec, adj_orig, edges_pos, edges_neg):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # predict on test set of edges
    adj_rec = rec
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(sigmoid(adj_orig[e[0], e[1]]))

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(sigmoid(adj_orig[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = np.array(linear_sum_assignment(w.max() - w)).T
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def clustering_evaluation(labels_true, labels):
    # logger.info("------------------------clustering result-----------------------------")
    # logger.info("original dataset length:{},pred dataset length:{}".format(
        # len(labels_true), len(labels)))
    # logger.info('number of clusters in dataset: %d' % len(set(labels_true)))
    # logger.info('number of clusters estimated: %d' % len(set(labels)))
    # logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # logger.info("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # logger.info("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # logger.info("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    # logger.info("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
    # logger.info("Purity Score: %0.3f" % purity_score(labels_true, labels))
    # logger.info("------------------------end ----------------------------------------")
    return metrics.homogeneity_score(labels_true, labels),\
            metrics.completeness_score(labels_true, labels), \
            metrics.v_measure_score(labels_true, labels), \
            metrics.adjusted_rand_score(labels_true, labels),\
            metrics.adjusted_mutual_info_score(labels_true, labels), \
            metrics.normalized_mutual_info_score(labels_true,labels), \
            purity_score(labels_true, labels),\
            f1_score(labels_true,labels,average='weighted'),\
            precision_score(labels_true,labels,average='weighted'),\
            recall_score(labels_true,labels,average='weighted')

def drop_feature(feature_matrix,delta):
    num_nodes, num_features = feature_matrix.shape
    mask = torch.tensor(np.random.binomial(1,delta,[num_nodes,1]))

    feature_matrix_dropped = feature_matrix*mask
    return feature_matrix_dropped


def drop_edge(adj,Y,delta=1):

    num_nodes, num_features = adj.shape

    # mask = torch.tensor(np.random.binomial(1,delta,[num_nodes,num_features]))

    for row in range(num_nodes):
        print(row)
        for col in range(num_nodes):
            if row!=col and adj[row,col]==1:
                if Y[row]!=Y[col]:
                    adj[row,col]=0
                    adj[col,row]=0

    print("after drop edge: edge number",adj.sum())
    return adj

def choose_cluster_votes(adj,prediction):

    n_nodes = adj.shape[0]

    new_prediction=[]
    for i in range(n_nodes):
        labels=prediction[(adj[i]>=1).tolist()]
        labels_max = Counter(labels)

        max_value = 0
        candicate_label =0
        for key,value in labels_max.items():
            if value > max_value:
                candicate_label = key
                max_value = value
        new_prediction.append(candicate_label)

    print("new prediction duplicate rate:",np.sum(np.array(new_prediction)==prediction)/len(prediction))
    return np.array(new_prediction)


def plot_tsne_non_centers(dataset,model_name,epoch,z,true_label,pred_label):

    tsne = TSNE(n_components=2, init='pca',perplexity=50.0)
    zs_tsne = tsne.fit_transform(z)

    cluster_labels=set(true_label)
    print(cluster_labels)
    index_group= [np.array(true_label)==y for y in cluster_labels]
    colors = cm.Set1(range(len(index_group)))

    fig, ax = plt.subplots(figsize=[5,5])
    for index,c in zip(index_group,colors):
        ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=30)
    # ax.legend(cluster_labels)

    # ax.scatter(zs_tsne[z.shape[0]:, 0], zs_tsne[z.shape[0]:, 1],marker='^',color='b',s=40)
    # plt.title('true label')
    # ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'true_label'))

    cluster_labels=set(pred_label)
    print(cluster_labels)
    index_group= [np.array(pred_label)==y for y in cluster_labels]
    colors = cm.tab10(range(len(index_group)))

    fig, ax = plt.subplots(figsize=[5,5])
    for index,c in zip(index_group,colors):
        ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=30)

    # for index,c in enumerate(colors):
        # ax.scatter(zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 0], zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 1],marker='^',color=c,s=40)

    ax.axis('off')
    # ax.legend(cluster_labels)
    # plt.title('pred label')
    # ax.legend()
    plt.tight_layout()
    plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'pred_label'))

def plot_tsne(dataset,model_name,epoch,z,mu_c,true_label,pred_label):

    tsne = TSNE(n_components=2, init='pca',perplexity=50.0)
    data = torch.cat([z,mu_c],dim=0).detach().numpy()
    zs_tsne = tsne.fit_transform(data)

    cluster_labels=set(true_label)
    print(cluster_labels)
    index_group= [np.array(true_label)==y for y in cluster_labels]
    colors = cm.Set1(range(len(index_group)))

    fig, ax = plt.subplots(figsize=[5,5])
    for index,c in zip(index_group,colors):
        ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=30)
    ax.axis('off')
    # ax.legend(cluster_labels)

    # ax.scatter(zs_tsne[z.shape[0]:, 0], zs_tsne[z.shape[0]:, 1],marker='^',color='b',s=40)
    # plt.title('true label')
    # ax.legend()
    plt.tight_layout()
    plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'true_label'))

    cluster_labels=set(pred_label)
    print(cluster_labels)
    index_group= [np.array(pred_label)==y for y in cluster_labels]
    colors = cm.tab10(range(len(index_group)))

    fig, ax = plt.subplots(figsize=[5,5])
    for index,c in zip(index_group,colors):
        ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=30)

    # for index,c in enumerate(colors):
        # ax.scatter(zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 0], zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 1],marker='^',color=c,s=40)

    # ax.legend(cluster_labels)
    ax.axis('off')
    # plt.title('pred label')
    # ax.legend()
    plt.tight_layout()
    plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'pred_label'))

def save_results(args,metrics_list):
    '''
    metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision]
    '''

    metrics_name=['H','C','V','Ari','Ami','Nmi','purity','accuracy','f1','precision','recall','entropy','time']
    wp = open('./result_logs/{}_{}_{}'.format(args.model,args.dataset,args.epochs),'a')
    wp.write("\n\n")
    if args.model =='gcn_vaece':
        wp.write("hidden1:{},hidden2:{},learning_rate:{},epochs:{},seed:{},beta:{}, omega:{}, mutual_loss:{}, clustering_loss:{}, using kmeans:{}, coembedding:{}\n".format(args.hidden1,args.hidden2,args.lr,args.epochs,args.seed,args.beta,args.omega,args.mutual_loss, args.clustering_loss, args.kmeans, args.coembedding))
    else:
        wp.write("hidden1:{},hidden2:{},learning_rate:{},epochs:{},seed:{}\n".format(args.hidden1,args.hidden2,args.lr,args.epochs,args.seed))

    for index,metric in enumerate(metrics_list):
        wp.write("{}\t".format(metrics_name[index]))
        for value in metric:
            wp.write("{}\t".format(value))
        wp.write("{}\t".format(round(np.mean(metric),4)))
        wp.write("{}\n".format(round(np.std(metric),4)))

    wp.write("mean list for latex table\n")
    wp.write("'Nmi','purity','Ari','f1','precision','recall','entropy'\n")
    for metric in ['Nmi','purity','Ari','f1','precision','recall','entropy']:
        for index, temp_metric in enumerate(metrics_name):
            if metric == temp_metric:
                wp.write("{} &".format(round(np.mean(sorted(metrics_list[index],reverse=True)[0:10]),4)))
    wp.write("\n")
    wp.close()


def entropy_metric(tru,pre):

    size = len(tru)
    unique_labels = len(set(tru))
    tru_d = []
    pre_d = []
    tru_s= Counter(tru)
    pre_s = Counter(pre)
    print(tru_s)
    print(pre_s)

    for i in range(unique_labels):
        tru_d.append(tru_s[i]/size)
        pre_d.append( pre_s[i]/size)

    print("label distribution for entropy")
    print('true labels:',tru_d)
    print('pred labels:',pre_d)

    return entropy(tru_d,pre_d)




