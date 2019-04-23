
# Initialize Parameters
K = 200
L = 100 
M = 60
N = 30


w1 = tf.Variable(tf.truncated_normal([28*28, K] ,stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))



y1 = tf.nn.sigmoid(tf.matmul(x,  w1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
y  = tf.nn.softmax(tf.matmul(y4, w5) + b5)



# ReLU activative function 

Y = tf.nn.relu(tf.matmul(X, W) + b)

# 在深度较深的模型中， 不适合sigmoid激活函数，它将所有的值量化到[0, 1]
# 易造成梯度消失.
