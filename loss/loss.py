import tensorflow as tf


# global_pool layer
net = slim.avg_pool2d(net, kernel_size)
# [N, 1, 1, C]
net = slim.dropout(net, keep_porb)
# [N, 1, 1, C]
net = slim.flatten(net, scope="PreLogitsFlatten")
# [N, C]
logits = slim.flatten(net, num_classes, activation=None, scope='Logits')
# logits.shape : [N, num_classes]
predictions = tf.nn.softmax(logits, name='Predictions')


def loss(logits, labels):
    regularization_loss = tf.reduce_sum(tf.get_collection(
        tf.GraphKeys,REGULARIZATION_LOSSES))
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, scope='Loss')
    
    loss = regularization_loss + cross_entropy_loss
    
    # Importment for Batch Normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    out_argmax = tf.argmax(tf.nn.softmax(logits), axis=-1, ouput_type=tf.int64)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(true_label, out_argmax), tf.float32))
