import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, weight_layer

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) ## [B,N,3] xyz
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3]) ## [B,N,6-3] normal

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    ## [B,512,64]
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    ## [B,128,256]
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    ## [B,1,1024]

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    ## [B,128,256]
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    ## [B,512,128]
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')
    ## [B,1,128]

    ## weight sum module
    w1_points, _ = weight_layer(l0_xyz, l0_points, 128, idx=None, is_training=is_training, scope='we1')
    w2_points, _ = weight_layer(l0_xyz, w1_points, 128, idx=None, is_training=is_training, scope='we2')

    # FC layers
    net = tf_util.conv1d(w2_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    BATCH_SIZE = 4
    NUM_POINT = 2048
    pts = np.random.rand(BATCH_SIZE, NUM_POINT, 6)
    labels = np.random.randint(0,49,size=[BATCH_SIZE, NUM_POINT])
    print pts.shape

    with tf.Graph().as_default():
        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 6))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        xyz = tf.slice(pointclouds_pl, [0, 0, 0], [-1, -1, 3])
        points = pointclouds_pl

        weight_pts, _ = get_model(points, is_training=is_training_pl)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init,{is_training_pl: True})

        dict = {pointclouds_pl: pts,is_training_pl:True}
        [weight_pts] = sess.run([weight_pts], feed_dict=dict)
        print weight_pts.shape
