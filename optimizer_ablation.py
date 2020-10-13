import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS



class Optimizer(object):
    def __init__(self, preds, labels, model, num_nodes,
                 global_step, new_learning_rate,
                 if_drop_edge = True, **kwargs):
        allowed_kwargs = {'placeholders'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        placeholders = kwargs.get("placeholders")
        noised_indexes = placeholders["noised_mask"]
        clean_indexes = placeholders["clean_mask"]
        en_preds_sub = preds
        en_labels_sub = labels
        self.ratio_loss_fea = FLAGS.ratio_loss_fea
        self.opt_op = 0  # this is the minimize function
        self.cost = 0  # this is the loss
        self.accuracy = 0  # this is the accuracy
        self.G_comm_loss = 0
        self.G_comm_loss_KL = 0
        self.D_loss = 0
        self.num_nodes = num_nodes
        self.if_drop_edge = if_drop_edge
        self.last_reg = tf.Variable(0,name = "last_reg", dtype = tf.float32, trainable=False)
        self.generate_optimizer = tf.train.RMSPropOptimizer(learning_rate= new_learning_rate)
        self.discriminate_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        generate_varlist = [var for var in tf.trainable_variables() if (
                    'generate' in var.name) or ('encoder' in var.name)]  
        # discriminate_varlist = [var for var in tf.trainable_variables() if 'discriminate' in var.name]
        if if_drop_edge == True:
            self.G_loss_ablation = self.loss_cross_entropy_logits_ablation(model)
            self.G_min_op = self.generate_optimizer.minimize(self.G_loss_ablation, global_step=global_step,var_list=generate_varlist)
            # self.D_loss = self.dis_cutmin_loss_clean_feature(model)
            # self.D_min_op = self.discriminate_optimizer.minimize(self.D_loss, global_step = global_step,
            #                                                      var_list = discriminate_varlist)


    def loss_cross_entropy_logits_ablation(self, model):
        adj_ori = model.adj_ori_dense - \
            tf.matrix_diag(tf.diag_part(model.adj_ori_dense))
        model_pred = model.x_tilde - tf.matrix_diag(tf.diag_part(model.x_tilde))
        loss_ablation = tf.nn.sigmoid_cross_entropy_with_logits(labels = adj_ori, logits = model_pred)
        G_loss_ablation = tf.reduce_mean(loss_ablation)
        ############### the feature loss part
        # indices = clean_indexes_2d
        # indices = tf.cast(indices, tf.int64)
        # values = tf.ones_like(clean_indexes, dtype = tf.float32)
        # shape = [self.num_nodes, self.num_nodes]
        # adj_in_comm = tf.SparseTensor(indices, values, shape)
        # self.score = tf.matmul(tf.sparse_tensor_dense_matmul(adj_in_comm, model.new_feature_prob),
        #                   tf.transpose(model.new_feature_prob))
        # losses_feature =(-1) *tf.log(tf.sigmoid(tf.trace(self.score)))
        ###############
        return G_loss_ablation

    pass

