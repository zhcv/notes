import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

print "\n\033[1;35m================================================================================\033[0m\n"
print 'Found GPU at: {}'.format(device_name)


# See https://www.tensorflow.org/turorials/using_gpu#allowing_gpu_memory_growth

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100, 100, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# Test execution onces to detect errors early.
try:
    sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
    print '''
        \n\nthis error most likely means that this notebook is not configured
        to use a GPU. Change this in notebook Settings via the command palette 
        (cmd/ctrl-shift-P) or the Edit menu.\n\n
        '''
    raise

def cpu():
    sess.run(net_cpu)

def gpu():
    sess.run(net_gpu)

# Runs the op several times
print '''Time (s) to vonvolve 32x7x7x3 filter over random 100x100x3 image 
    batch x height x width x channel. Sum of ten runs.'''

print 'CPU (s):'
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print cpu_time

print 'GPU (s):'
gpu_time = timeit.timeit('gpu()', number=100, setup="from __main__ import gpu")
print gpu_time

print 'GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time))
print "\n\033[1;35m================================================================================\033[0m\n"

sess.close()
