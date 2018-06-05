import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X , y,test_size  = 0.3)

input_size = 64
output_size = 10
norm = False
hidden_units1 = 100
hidden_units2 = 50

def fc_layer_with_bn(x_input, num_units, activation, is_training):
    layer = tf.layers.dense(x_input, num_units, use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = activation(layer)
    return layer



def compute_accuracy_bn(v_xs, v_ys):
    global prediction #先把prediction定义为全局变量

    y_pre = sess.run(prediction, feed_dict={xs:v_xs, on_train:False}) #生成预测值（概率），10分类，所以一个样本是10列概率
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) #比较概率最大值的位置和真实标签位置是否一样，一样就是true
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #计算平均正确率
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys}) #生成正确率
    return result

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None,input_size], name = "x_input") # 输入数据：None不限制样本数，64=8x8,每个样本64个像素点（特征）
    ys = tf.placeholder(tf.float32, [None,output_size], name = "y_input") # 输出数据：None不限制样本数，10个输出（10分类问题）

on_train = tf.placeholder(tf.bool, name = "on_train")


#（隐藏层）：输入为上一层输出=64
l1 = fc_layer_with_bn(xs, hidden_units1, activation=tf.nn.relu, is_training=on_train)
d1 = tf.layers.dropout(l1, rate = 0.5, training=on_train)
l2 = fc_layer_with_bn(d1, hidden_units2, activation=tf.nn.relu, is_training=on_train)
d2 = tf.layers.dropout(l2, rate = 0.3, training=on_train)
#（输出层）：输入为上一层输出=100，输出10
prediction = fc_layer_with_bn(d2, output_size, activation=tf.nn.softmax, is_training=on_train)

with tf.name_scope("cross_entropy_bn"):
    cross_entropy =  tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)
    tf.summary.scalar("cross_entropy_bn",cross_entropy)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope("train"):
        train_step = tf. train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    # 将上述可视化合并
    merged = tf.summary.merge_all()

    # 将以上结构写入文件，分为train和test
    train_bn_writer = tf.summary.FileWriter("/home/kenn/tensorflow_learning/logs/DNN/train",sess.graph)
    test_bn_writer = tf.summary.FileWriter("/home/kenn/tensorflow_learning/logs/DNN/test",sess.graph)
    # 终端激活对应环境后执行 tensorboard --logdi'file:///home/kenn/tensorflow_learning/logs'
    # localhost:6006看
    
    init = tf.global_variables_initializer()

    sess.run(init)
    for i in range(400):
        #通过placeholder定义输入的话，都要用feed_dict载入数据，dropout参数也要从这里输入
        sess.run(train_step, feed_dict = {xs:X_train, ys:y_train, on_train:True})

        if i%5 == 0:
            print(compute_accuracy_bn(X_test,y_test))

            #预测的时候droprate为0
            train_result = sess.run(merged, feed_dict = {xs:X_train, 
                                                         ys:y_train, 
                                                         on_train:False})
            test_result = sess.run(merged, feed_dict = {xs:X_test, 
                                                        ys:y_test, 
                                                        on_train:False})

            train_bn_writer.add_summary(train_result,i)
            test_bn_writer.add_summary(test_result,i)

