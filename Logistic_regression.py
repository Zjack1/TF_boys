import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels
print trainimg.shape#(55000, 784)

x=tf.placeholder("float",[None,784])#[[1,2,45,67,33....]....]
y=tf.placeholder("float",[None,10])#[[0,0,0,0,0,0,1,0,0,0,0],...]
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
###model:
actv=tf.nn.softmax(tf.matmul(x,W)+b)#10 score
###cost_function(softmax_loss):
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))
###optimize(SGD):
learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#prediction
pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))#max_suoying_same_ture
#accuracy
accr=tf.reduce_mean(tf.cast(pred,"float"))#pred-->"float"
init=tf.global_variables_initializer()


training_epochs=50
batch_size=100
display_step=5
sess=tf.Session()
sess.run(init)
####minibatch_learning
for epoch in  range(training_epochs):
	avg_cost=0
	num_batch=int(mnist.train.num_examples/batch_size)
	for i in range(num_batch):
		batch_xs,batch_ys=mnist.train.next_batch(batch_size)
		sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
		feeds={x:batch_xs,y:batch_ys}
		avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch
	if epoch%display_step==0:
		feeds_train={x:batch_xs,y:batch_ys}
		feeds_test={x:mnist.test.images,y:mnist.test.labels}
		train_acc=sess.run(accr,feed_dict=feeds_train)
		test_acc=sess.run(accr,feed_dict=feeds_test)
		print ("Epoch:%03d/%03d cost:%.9f train_acc:%.3f test_acc:%.3f"%(epoch,training_epochs,avg_cost,train_acc,test_acc))
print 'DONE!!!!!!!!!!!'
    








