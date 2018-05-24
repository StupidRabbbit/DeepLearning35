# coding=utf-8
#mnist手写数字识别 迭代20000次 训练集上准确率达到1.0 测试集准确率达99.18%
#dropout 0.5
#batch 50
import tensorflow as tf
#version=1.2.1
# print(tf.__version__)
from  tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('D:\dl_data\mnist_data',one_hot=True)
#55000
# print(mnist.train.num_examples)
#5000
# print(mnist.validation.num_examples)
#10000
# print(mnist.test.num_examples)
#取训练数据的一小部分作为下一个训练batch
batch_size=100
# print('Xshape',xs.shape)
# print('Yshape',ys.shape)
#mnist数据集相关的常数
#输入的784个像素和10个输出分类
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
#batch越小，越接近随机梯度下降，batch越大，越接近梯度下降
#数据量很大的时候随机梯度下降收敛速度更快
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
#衰减速率，学习速率每次都衰减到原来的0.99嗯是的。
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRANING_STEPS=3000
MOVING_AVERAGE_DECAY=0.99
# create model

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#为什么bias是常量 很奇怪
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#一会儿再倒回来看理论
#卷积和池化
#步长就是卷积核每次滑动的时候跨的长度
#做padding是因为想要保持和原来图像相同的尺寸，不让图像在训练的过程中
#缩小的太快
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#比较传统的2x2池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#None表示任意数量
#占位符：2维浮点数张量来表示每一张图，后面表示
x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y_=tf.placeholder('float',[None,10])
# y=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
#设置初始值

# 定义第一层卷积层变量，输入的图片深度为1，输出变为32，patch大小为5*5
#输出为32 即有32个卷积核，生成32个不同的图像
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

#-1为任意数量的图片，28*28=784，这里把784个像素摞起来了，最后一个1是代表深度，
#这时图片的深度只有1，还未经过卷积神经网络
x_image=tf.reshape(x,[-1,28,28,1])
#定义第一个卷积层的操作
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#28*28*1的图片

#到了第二层图片已经变成了14*14*32，经过了第一层的池化
#定义第二层卷积层变量
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
#定义第二层卷积层操作
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#到这里已经变成7*7*64的压缩图片了，深度为64，
#每一次第一个参数都是输入的图片的像素数量，第二个参数为输出像素数量
#这一步要把压缩图像的像素展成一维像素 1*1024
#定义全连接层的变量
#但是这块已经做了padding的呀
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#
#定义fc层后的drop_out，前面卷积都是在处理图片信息，这里才连接神经层

keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#定义输出卷积
#
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# w=tf.Variable(tf.zeros([784,10]))
# b=tf.Variable(tf.zeros([10]))
# y=tf.nn.softmax(tf.matmul(x,w)+b)
#设置优化参数
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
# train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correction_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correction_prediction,'float'))

#初始化所有变量，并且开启一个会话
# return a global_variables_initializer()
#global_variables_initializer().run()和下面这个效果是一样的
init=tf.initialize_all_variables()
# sess=tf.Session()
#可以更灵活地构建计算图，可以在运行过程中加入图
sess=tf.InteractiveSession()
sess.run(init)
#训练阶段
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0 and i!=0:
    #调用这个eval函数会产生这个张量的值
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print( 'step:',i,'accuracy:',train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print( 'test accuracy ',accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#训练阶段
# for i in range(TRANING_STEPS):
#     xs, ys = mnist.train.next_batch(batch_size)
#     sess.run(train_step,feed_dict={x:xs,y_:ys})
#
# correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
# print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))








