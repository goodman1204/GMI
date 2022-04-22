import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx

def load_AN(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    attri_file = open("data/{}.node".format(dataset), 'r')
    label_file = open("data/{}.label".format(dataset),'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    labels_raw = label_file.readlines()
    edge_file.close()
    attri_file.close()
    label_file.close()

    node_num = int(edges[0].split()[1].strip())
    edge_num = int(edges[1].split()[1].strip())
    attribute_number = int(attributes[1].split()[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    edge_num_no_selfloop= 0
    for line in edges:
        node1 = int(line.split()[0].strip())
        node2 = int(line.split()[1].strip())
        if node1==node2:
            continue
        adj_row.append(node1)
        adj_col.append(node2)
        edge_num_no_selfloop+=1
    adj = sp.csc_matrix((np.ones(edge_num_no_selfloop), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split()[0].strip())
        attribute1 = int(line.split()[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))

    labels=[]
    for label in labels_raw:
        label = int(label.strip())
        labels.append(label)
    return adj, attribute, np.array(labels)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    return adj, features, labels, idx_train, idx_val, idx_test

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, features_orig, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['features_orig']: features_orig})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_edges_false) < num_test :
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val :
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val :
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test :
                    test_edges_false.append([i, j])

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    print('sum adj_train',adj_train.sum()/2)
    adj_train = adj_train + adj_train.T
    # adj_train = adj_train
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_feas(features):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 30.))
    num_val = int(np.floor(len(feas) / 30.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_feas_false) < num_test :
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val :
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val :
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test :
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false

