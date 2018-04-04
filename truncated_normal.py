import tensorflow as tf
initial = tf.truncated_normal(shape=[3,3], mean=0, stddev=1)
print(tf.Session().run(initial))
#产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]