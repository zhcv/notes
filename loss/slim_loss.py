"""Fork from
https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py
"""
import tensorflow as tf

slim = tf.contrib.slim



# Create a dataset provider that loads data from the dataset #
with tf.device(deploy_config.inputs_device()):
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=FLAGS.num_readers,
      common_queue_capacity=20 * FLAGS.batch_size,
      common_queue_min=10 * FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])
  label -= FLAGS.labels_offset

  train_image_size = FLAGS.train_image_size or network_fn.default_image_size

  image = image_preprocessing_fn(image, train_image_size, train_image_size)

  images, labels = tf.train.batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_preprocessing_threads,
      capacity=5 * FLAGS.batch_size)
  labels = slim.one_hot_encoding(
      labels, dataset.num_classes - FLAGS.labels_offset)
  batch_queue = slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity=2 * deploy_config.num_clones)


def clone_fn(batch_queue):
    """Allows data parallelism by creating multiple clones of network_fn."""
    images, labels = batch_queue.dequeue()
    logits, end_points = network_fn(images)

    # Specify the loss in end_points
    if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AugLogits'], labels,
            label_smoothing=label_smoothing, weights=1.0)

    slim.losses.softmax_cross_entropy(
        logits, labels, label_smoothing=label_smoothing, weights=1.0)
    return end_points
