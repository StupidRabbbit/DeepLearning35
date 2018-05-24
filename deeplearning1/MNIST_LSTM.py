# coding=utf-8
#LSTM来做的MNIST手写识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('D:\dl_data\mnist_data',one_hot=True)

#hyperparameters
LEARNING_RATE=0.001
TRAINING_ITERS=100000
BATCH_SIZE=128

#图片 28*28 input是行像素 step是列像素
#timestep是时序步数，每一次预测需要输入28行
n_inputs=28
n_steps=28
#既然是定义一层隐藏层 那就不只是一个神经元 应该是一层有128个神经元
#一般都是64的倍数，64，128，256 blabla
n_hidden_units=128
n_classes=10

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

#Define Weights
weights={
    # (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases={
    #(10,)
    #一维数组，一行n列 是对的 一维数组加到二维数组中相当于在二维数组的每一行都加上一个一维数组
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))

}
def RNN(X,weights,biases):
    #hidden layer for input cell
    #为什么hidden layer是128层？
    #X(128batch,28 steps,28 inputs)
    #==>(128*28,28)
    #为了支持二维矩阵乘法
    #这次乘的是输入到隐藏层的连接权
    X=tf.reshape(X,[-1,n_inputs])
    X_in=tf.matmul(X,weights['in'])+biases['in']
    #-1是batch size
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    # 理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免梯度消散的问题，会规定一个最大的序列长度
    #max_step在这里就为steps
    #cell
    #state_is_tuple state就是cell里面的计算结果
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #lstm 被分成两个部分 一个状态元组（c_state,m_state) 主线状态和分线状态,RNN只有m
    #如果一次输入一行28个像素 一个时序输入28步（行） 那么就可以理解 为什么会有BATCH_SIZE个状态了
    #会为BATCH_SIZE中每个样本生成一组状态列表，状态数量与timestep有关，这个里面是与28有关
    #所以输出LSTM神经层的输出就是output[batch_size,-1,hidden_units]中间那个是timestep
    #时序是指timestep的时序，每一个样本都会有一个时序状态列表
    _init_state=lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)

    #state一刚开始是0，后面慢慢的累积，慢慢的有记忆
    #time major？？没懂 和维度有关 如果输入的第一个参数为Time-step 则为true
    #outputs 就是一个list 会不停往里面加东西（每一次的输出）, state是神经元里面的状态
    # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    outputs,state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
    #这里并没有
    results=tf.matmul(state[1],weights['out'])+biases['out']

    #state[1]在这里面和outputs[-1] 最后一个输出一样 别的里面不一定一样
    #为什么会一样，因为只有一个隐藏层，第一层输出和状态就是一样的？？
    #一般都是用output[-1]
    #hidden layer for output as the final results
    return results

pred=RNN(x,weights,biases)
"""tf.nn.softmax_cross_entropy_with_ligits:Computes softmax cross entropy between `logits` and `labels`."""
#logits指的是没经过softmax的概率 一般是输出层的输出，所以之前是写反了 label是真实标签
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
#减小交叉熵
tf.unstack()
train_op=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*BATCH_SIZE<TRAINING_ITERS:
        batch_xs,batch_ys=mnist.train.next_batch(BATCH_SIZE)
        # print(batch_xs.shape,'original batch_xs')
        batch_xs=batch_xs.reshape([BATCH_SIZE,n_steps,n_inputs])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        if step%20==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1

writer=tf.train.SummaryWriter('/path/to/log',tf.get_default_graph())
writer.close()





