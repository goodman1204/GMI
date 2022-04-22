import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy.sparse as sp
from models import GMI, LogReg
from utils import process

import time
from preprocessing import mask_test_feas,mask_test_edges, load_AN, check_symmetric,load_data
from util_new import preprocess_graph, get_roc_score, sparse_to_tuple,sparse_mx_to_torch_sparse_tensor,cluster_acc,clustering_evaluation, find_motif,drop_feature, drop_edge,choose_cluster_votes,plot_tsne,save_results,entropy_metric,plot_tsne_non_centers
from evaluation import clustering_latent_space
from hungrian import label_mapping
from collections import Counter

"""command-line interface"""
parser = argparse.ArgumentParser(description="PyTorch Implementation of GMI")
parser.add_argument('--dataset', default='cora',
                    help='name of dataset. if on citeseer and pubmed, the encoder is 1-layer GCN. you need to modify gmi.py')
parser.add_argument('--gpu', type=int, default=0,
                    help='set GPU')
"""training params"""
parser.add_argument('--hid_units', type=int, default=32,
                    help='dim of node embedding (default: 512)')
parser.add_argument('--model', type=str, default='GMI')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 550)')
parser.add_argument('--epoch_flag', type=int, default=20,
                    help=' early stopping (default: 20)')
parser.add_argument('--num_run',type=int,default=1,help='Number of running times')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--l2_coef', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--negative_num', type=int, default=5,
                    help='number of negative examples used in the discriminator (default: 5)')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='parameter for I(w_ij; a_ij) (default: 1.0)')
parser.add_argument('--activation', default='prelu',
                    help='activation function')

parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--synthetic_num_nodes',type=int,default=1000)
parser.add_argument('--synthetic_density', type=float, default=0.1)

parser.add_argument('--nClusters',type=int,default=7)
parser.add_argument('--cuda', type=int, default=0, help='training with GPU.')
args, unknown = parser.parse_known_args()
###############################################
# This section of code adapted from Petar Veličković/DGI #
###############################################

args = parser.parse_args()
torch.cuda.set_device(args.gpu)

print('Loading', args.dataset)
# adj_ori, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
# features, _ = process.preprocess_features(features)

if args.dataset in ['cora','pubmed','citeseer']:
    adj_ori, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    Y = np.argmax(labels,1) # labels is in one-hot format
elif args.dataset in ['Flickr','BlogCatalog']:
    adj_ori, features, Y= load_AN(args.dataset)
else:
    adj_ori, features, Y= load_AN("synthetic_{}_{}".format(args.synthetic_num_nodes,args.synthetic_density))


labels = Y
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[0]
adj = process.normalize_adj(adj_ori + sp.eye(adj_ori.shape[0]))

sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
features = torch.FloatTensor(features.toarray()[np.newaxis])
labels = torch.FloatTensor(labels)

model = GMI(ft_size, args.hid_units, args.activation)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

if torch.cuda.is_available():
    print('GPU available: Using CUDA')
    model.cuda()
    features = features.cuda()
    sp_adj = sp_adj.cuda()
    labels = labels.cuda()

xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

adj_dense = adj_ori.toarray()
adj_target = adj_dense+np.eye(adj_dense.shape[0])
adj_row_avg = 1.0/np.sum(adj_dense, axis=1)
adj_row_avg[np.isnan(adj_row_avg)] = 0.0
adj_row_avg[np.isinf(adj_row_avg)] = 0.0
adj_dense = adj_dense*1.0
for i in range(adj_ori.shape[0]):
    adj_dense[i] = adj_dense[i]*adj_row_avg[i]
adj_ori = sp.csr_matrix(adj_dense, dtype=np.float32)


start_time = time.time()

for epoch in range(args.epochs):
    model.train()
    optimiser.zero_grad()

    res = model(features, adj_ori, args.negative_num, sp_adj, None, None)

    loss = args.alpha*process.mi_loss_jsd(res[0], res[1]) + args.beta*process.mi_loss_jsd(res[2], res[3]) + args.gamma*process.reconstruct_loss(res[4], adj_target)
    print('Epoch:', (epoch+1), '  Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_gmi_{}_{}.pkl'.format(args.dataset,args.epochs))
    else:
        cnt_wait += 1

    if cnt_wait == args.epoch_flag:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t+1))
model.load_state_dict(torch.load('best_gmi_{}_{}.pkl'.format(args.dataset,args.epochs)))

z = model.embed(features, sp_adj)
z = z.cpu().numpy()
z = z.squeeze()

end_time = time.time()


mean_h=[]
mean_c=[]
mean_v=[]
mean_ari=[]
mean_ami=[]
mean_nmi=[]
mean_purity=[]
mean_accuracy=[]
mean_f1=[]
mean_precision=[]
mean_recall = []
mean_entropy = []
mean_time= []



tru=Y


pre,mu_c=clustering_latent_space(z,tru)
plot_tsne_non_centers(args.dataset,args.model,args.epochs,z,Y,pre)

for i in range(args.num_run):
    pre,mu_c=clustering_latent_space(z,tru)


    pre = label_mapping(tru,pre)
    H, C, V, ari, ami, nmi, purity,f1_score,precision,recall= clustering_evaluation(tru,pre)

    entropy = entropy_metric(tru,pre)

    acc = cluster_acc(pre,tru)[0]
    mean_h.append(round(H,4))
    mean_c.append(round(C,4))
    mean_v.append(round(V,4))
    mean_ari.append(round(ari,4))
    mean_ami.append(round(ami,4))
    mean_nmi.append(round(nmi,4))
    mean_purity.append(round(purity,4))
    mean_accuracy.append(round(acc,4))
    mean_f1.append(round(f1_score,4))
    mean_precision.append(round(precision,4))
    mean_recall.append(round(recall,4))
    mean_entropy.append(round(entropy,4))
    mean_time.append(round(end_time-start_time,4))

# metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision,mean_recall,mean_entropy]
metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision,mean_recall,mean_entropy,mean_time]
save_results(args,metrics_list)

###### Report Final Results ######
print('Homogeneity:{}\t mean:{}\t std:{}\n'.format(mean_h,round(np.mean(mean_h),4),round(np.std(mean_h),4)))
print('Completeness:{}\t mean:{}\t std:{}\n'.format(mean_c,round(np.mean(mean_c),4),round(np.std(mean_c),4)))
print('V_measure_score:{}\t mean:{}\t std:{}\n'.format(mean_v,round(np.mean(mean_v),4),round(np.std(mean_v),4)))
print('adjusted Rand Score:{}\t mean:{}\t std:{}\n'.format(mean_ari,round(np.mean(mean_ari),4),round(np.std(mean_ari),4)))
print('adjusted Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_ami,round(np.mean(mean_ami),4),round(np.std(mean_ami),4)))
print('Normalized Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_nmi,round(np.mean(mean_nmi),4),round(np.std(mean_nmi),4)))
print('Purity:{}\t mean:{}\t std:{}\n'.format(mean_purity,round(np.mean(mean_purity),4),round(np.std(mean_purity),4)))
print('Accuracy:{}\t mean:{}\t std:{}\n'.format(mean_accuracy,round(np.mean(mean_accuracy),4),round(np.std(mean_accuracy),4)))
print('F1-score:{}\t mean:{}\t std:{}\n'.format(mean_f1,round(np.mean(mean_f1),4),round(np.std(mean_f1),4)))
print('precision_score:{}\t mean:{}\t std:{}\n'.format(mean_precision,round(np.mean(mean_precision),4),round(np.std(mean_precision),4)))
print('recall_score:{}\t mean:{}\t std:{}\n'.format(mean_recall,round(np.mean(mean_recall),4),round(np.std(mean_recall),4)))
print('entropy:{}\t mean:{}\t std:{}\n'.format(mean_entropy,round(np.mean(mean_entropy),4),round(np.std(mean_entropy),4)))
print("True label distribution:{}".format(tru))
print(Counter(tru))
print("Predicted label distribution:{}".format(pre))
print(Counter(pre))
