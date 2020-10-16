import tensorflow as tf
import random
import tensorflow.contrib.slim as slim
from utils import randomly_add_edges, randomly_delete_edges, randomly_flip_features,flip_features_fix_attr
from utils import add_edges_between_labels, denoise_ratio, get_noised_indexes, load_data_subgraphs
from utils import PSNR, WL,WL_no_label
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import datetime
import numpy as np
import scipy.sparse as sp
import time
import os
seed = 152   
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges,get_target_nodes_and_comm_labels, construct_feed_dict_trained
from gvae_ablation import mask_gvae
from optimizer_ablation import Optimizer
from gcn.utils import load_data
from tqdm import tqdm
from ND import ND
import scipy.io as scio
from gcn import train_test as GCN
from graph.dataset import load
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
##### this is for gae part
flags.DEFINE_integer('n_clusters', 6, 'Number of epochs to train.')    # this one can be calculated according to labels
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
####### for clean gcn training and test
flags.DEFINE_float('gcn_learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('gcn_hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('gcn_weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for: early stopping (# of epochs).')
###########################
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('mincut_r', 0.3, 'The r parameters for the cutmin loss orth loss')
flags.DEFINE_string('model', 'mask_gvae', 'Model string.')
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
from tensorflow.python.client import device_lib
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("latent_dim" , 16, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 1e-02, "the init of learn rate")
flags.DEFINE_integer("k", 20, "The edges to delete for the model")
flags.DEFINE_integer("k_noise", 20, "The k edges to add noise")
flags.DEFINE_integer("k_features", 300, "The nodes to flip features for the model")
flags.DEFINE_integer("k_features_noise", 300, "The nodes to add noise and flip features")
flags.DEFINE_integer("k_features_dim", 1, "The nodes to add noise and flip features")
flags.DEFINE_float('ratio_loss_fea', 1, 'the ratio of generate loss for features')
flags.DEFINE_boolean("train", True, "Training or Test")
###############################
if_train = FLAGS.train
cv_index = int(if_train)
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
###################################
### read and process the graph
model_str = FLAGS.model
dataset_str = FLAGS.dataset
noise_ratio = 0.1
## Load datasets
# IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, MUTAG, PTC_MR
dataset_index = "IMDB-BINARY"
train_structure_input, train_feature_input, train_y, \
    train_num_nodes_all, test_structure_input, test_feature_input, \
    test_y, test_num_nodes_all = load_data_subgraphs(dataset_index, train_ratio=0.9)
##
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj_norm, adj_norm_sparse = preprocess_graph(adj)

n_class = y_train.shape[1]
features_normlize = normalize(features.tocsr(), axis=0, norm='max')
features = sp.csr_matrix(features_normlize)

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless
# Some preprocessing

num_nodes = adj.shape[0]
features_csr = features
features_csr = features_csr.astype('float32')
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
gpu_id = 1

# Create model
cost_val = []
acc_val = []
val_roc_score = []

def get_new_adj(feed_dict, sess, model):
    new_adj = model.new_adj.eval(session=sess, feed_dict=feed_dict)
    new_adj = new_adj - np.diag(np.diagonal(new_adj))
    return new_adj

def add_noises_on_adjs(adj_list, num_nodes, noise_ratio = 0.1, ):
    noised_adj_list = []
    # add_idx_list = []
    adj_orig_list = []
    k_list = []
    for i in range(len(adj_list)):
        adj_orig = adj_list[i]
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                            shape=adj_orig.shape)  # delete self loop
        adj_orig.eliminate_zeros()
        # adj_new, add_idxes = add_edges_between_labels(adj_orig, int(noise_ratio* num_nodes[i]), y_train)
        adj_new,k_real = randomly_add_edges(adj_orig, int(noise_ratio* adj_orig[:num_nodes[i], :num_nodes[i]].sum() / 2), num_nodes[i])
        k_list.append(k_real)
        noised_adj_list.append(adj_new)
        # add_idx_list.append(add_idxes)
        adj_orig_list.append(adj_orig)
    return noised_adj_list, adj_orig_list, k_list

def get_new_feature(feed_dict, sess,flip_features_csr, feature_entry, model):
    new_indexes = model.flip_feature_indexes.eval(session = sess, feed_dict = feed_dict)
    flip_features_lil = flip_features_csr.tolil()
    for index in new_indexes:
        for j in feature_entry:
            flip_features_lil[index, j] = 1 - flip_features_lil[index, j]
    return flip_features_lil.tocsr()
# Train model

def save_noise_mat():
    train_adj_list, train_adj_orig_list , train_k_list= add_noises_on_adjs(train_structure_input, train_num_nodes_all)
    test_adj_list, test_adj_orig_list , test_k_list = add_noises_on_adjs(test_structure_input, test_num_nodes_all)
    save_path = "./data/NE/"
    for i in range(len(test_adj_list)):
        file_name = dataset_index + "_" +str(i) +".mat"
        scio.savemat(os.path.join(save_path, file_name), {'adj_new': test_adj_list[i].toarray()})
    return



def train():
    ## add noise label
    train_adj_list, train_adj_orig_list, train_k_list = add_noises_on_adjs(train_structure_input, train_num_nodes_all)
    test_adj_list, test_adj_orig_list, test_k_list = add_noises_on_adjs(test_structure_input, test_num_nodes_all)

    adj = train_adj_list[0]
    features_csr = train_feature_input[0]
    features = sparse_to_tuple(features_csr.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_adj_orig_list[0]
    adj_label = train_adj_list[0] + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    num_nodes = adj.shape[0]
    ############
    psnr_list = []
    wls_list = []
    load_path = "./data/NE_denoise"
    for i in range(len(test_feature_input)):
        load_file = os.path.join(load_path,  dataset_index + "_" +str(i) +".mat")
        psnr, wls = test_one_graph_NE(test_adj_list[i], test_adj_orig_list[i],test_feature_input[i],test_num_nodes_all[i], load_file)
        psnr_list.append(psnr)
        wls_list.append(wls)
    # new_adj = get_new_adj(feed_dict,sess, model)
    ##################################
    ################## the PSRN and WL #########################
    print("#"*15)
    print("The PSNR is:")
    psnr_list = [x for x in psnr_list if x != float("inf")] ## here isa bug, we can not check it
    print(np.mean(psnr_list))
    print("The WL is :")
    print(np.mean(wls_list))
    return np.mean(psnr_list),np.mean(wls_list)

def train_one_graph(adj,adj_orig, features_csr ,num_node ,model, opt,placeholders, sess,new_learning_rate,feed_dict, epoch, graph_index):
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    adj_new  = adj
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    # L = row_sum - adj_new
    # ori_Lap = features_csr.transpose().dot(L).dot(features_csr)
    # ori_Lap_trace = ori_Lap.diagonal().sum()
    # ori_Lap_log = np.log(ori_Lap_trace)
    features = sparse_to_tuple(features_csr.tocoo())
    adj_norm, adj_norm_sparse = preprocess_graph(adj_new)
    adj_norm_sparse_csr = adj_norm_sparse.tocsr()
    adj_label = adj_new + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    ############
    ## set the placeholders
    # build models
    adj_clean = adj_orig.tocoo()
    adj_clean_tensor = tf.SparseTensor(indices =np.stack([adj_clean.row,adj_clean.col], axis = -1),
                                       values = adj_clean.data, dense_shape = adj_clean.shape )
    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    ### initial clean and noised_mask
    clean_mask = np.array([1,2,3,4,5])
    noised_mask = np.array([6,7,8,9,10])
    noised_num = noised_mask.shape[0] / 2
    ##################################
    #
    feed_dict.update({placeholders["adj"]: adj_norm})
    feed_dict.update({placeholders["adj_orig"]: adj_label})
    feed_dict.update({placeholders["features"]: features})
    node_mask = np.ones([adj.shape[0], n_class])
    node_mask[num_node:, :] = 0
    feed_dict.update({placeholders['node_mask']: node_mask})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    model.k = adj_new.sum()*noise_ratio
    #####################################################
    t = time.time()
    ########
    # last_reg = current_reg
    if epoch >= 0:  ## here we can contorl the manner of new model
        _= sess.run([opt.G_min_op], feed_dict=feed_dict,options=run_options)

    else:
        _, x_tilde = sess.run([opt.D_min_op, model.realD_tilde], feed_dict = feed_dict, options=run_options)
        if epoch == int(FLAGS.epochs / 2):
            noised_indexes, clean_indexes = get_noised_indexes(x_tilde, adj_new, num_node)
            feed_dict.update({placeholders["noised_mask"]: noised_indexes})
            feed_dict.update({placeholders["clean_mask"]: clean_indexes})
            feed_dict.update({placeholders["noised_num"]: len(noised_indexes)/2})
    ##
    if epoch % 1 == 0 and graph_index == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "time=", "{:.5f}".format(time.time() - t))
        G_loss, new_learn_rate_value = sess.run([opt.G_loss_ablation,new_learning_rate],feed_dict=feed_dict,  options = run_options)
        print("Step: %d,G: loss=%.7f , LR=%.7f" % (epoch, G_loss, new_learn_rate_value))
        ##########################################
    return

def test_one_graph(adj , adj_orig, features_csr, num_node):
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    adj_new = adj
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    # L = row_sum - adj_new
    # ori_Lap = features_csr.transpose().dot(L).dot(features_csr)
    # ori_Lap_trace = ori_Lap.diagonal().sum()
    # ori_Lap_log = np.log(ori_Lap_trace)
    features = sparse_to_tuple(features_csr.tocoo())
    adj_label = adj_new + sp.eye(adj.shape[0])
    adj_label_sparse = adj_label
    adj_label = sparse_to_tuple(adj_label)
    adj_clean = adj_orig.tocsr()

    adj_norm, adj_norm_sparse = preprocess_graph(adj_new)
    # feed_dict = construct_feed_dict(adj_norm, adj_label, features, clean_mask, noised_mask, noised_num, placeholders)
    x_tilde = sess.run(model.realD_tilde, feed_dict=feed_dict, options=run_options)
    noised_indexes, clean_indexes = get_noised_indexes(x_tilde, adj_new, num_node)
    model.k = adj_new.sum() * noise_ratio
    new_adj = get_new_adj(feed_dict, sess, model)
    new_adj_sparse = sp.csr_matrix(new_adj)
    psnr = PSNR(adj_clean[:num_node, :num_node], new_adj_sparse[:num_node, :num_node])
    y_label = y_train + y_val + y_test
    wls = WL_no_label(adj_clean[:num_node, :num_node], new_adj_sparse[:num_node, :num_node])
    return psnr, wls

def test_one_graph_NE(adj , adj_orig, features_csr, num_node, load_file):
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    adj_new = adj
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    ##NE algorithm
    # load_file = "noised_graph/cora_1800_900.mat"
    temp_data = scio.loadmat(load_file)
    new_adj = temp_data["denoised_adj"]
    new_adj = sp.csr_matrix(new_adj)
    adj_clean = adj_orig.tocsr()
    psnr = PSNR(adj_clean[:num_node, :num_node], new_adj[:num_node, :num_node])
    y_label = y_train + y_val + y_test
    wls = WL_no_label(adj_clean[:num_node, :num_node], new_adj[:num_node, :num_node])
    return psnr, wls



FLAGS = flags.FLAGS
if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    ######################################### step 1
    # save_noise_mat()
    ######################################### step 2 load the denoise matrix in the status
    with open("results/results_%d_%s.txt"%(FLAGS.k, current_time), 'w+') as f_out:
        f_out.write('PSNR'+ ' ' + 'WL' + "\n")
        for i in range(1):
            psnr,wls, = train()
            f_out.write(str(psnr)+ ' '+str(wls) + "\n")
    print(dataset_index)
    print(current_time)
    
