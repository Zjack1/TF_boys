# -*- coding: UTF-8 -*-
import numpy as np  
import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  
import random   

number = ['0','1','2','3','4','5','6','7','8','9']  
#alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
#ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  

#def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):  
def random_captcha_text(char_set=number, captcha_size=4):#定义这个验证码是有数字组成的
    captcha_text = []  
    for i in range(captcha_size):  
        c = random.choice(char_set)  
        captcha_text.append(c)  
    return captcha_text  #list中存放4个验证码数字
   
#生成训练用的验证码图片和标签
def gen_captcha_text_and_image():  
    image = ImageCaptcha() #将验证码变成图片 
   
    captcha_text = random_captcha_text()  
    captcha_text = ''.join(captcha_text)  #吧数字变成list（str类型）
   
    captcha = image.generate(captcha_text)  
    #image.write(captcha_text, captcha_text + '.jpg')   
   
    captcha_image = Image.open(captcha) #把list中的str转换成图片
    captcha_image = np.array(captcha_image)#把图片转换成我们网络可以读取的arry类型
    return captcha_text, captcha_image  

def convert2gray(img):#将彩色图片转成灰度图
    if len(img.shape) > 2:  
        gray = np.mean(img, -1)  
        # 上面的转法较快，正规转法如下  
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]  
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
        return gray  
    else:  
        return img  

   
  
def text2vec(text):#文本转成向量
    text_len = len(text)  
    if text_len > MAX_CAPTCHA:  
        raise ValueError('验证码最长4个字符')  
   
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)  
    for i, c in enumerate(text):  
        idx = i * CHAR_SET_LEN + int(c)  
        vector[idx] = 1  
    return vector  

def vec2text(vec):  # 向量转回文本
    """
    char_pos = vec.nonzero()[0]  
    text=[]  
    for i, c in enumerate(char_pos):  
        char_at_pos = i #c/63  
        char_idx = c % CHAR_SET_LEN  
        if char_idx < 10:  
            char_code = char_idx + ord('0')  
        elif char_idx <36:  
            char_code = char_idx - 10 + ord('A')  
        elif char_idx < 62:  
            char_code = char_idx-  36 + ord('a')  
        elif char_idx == 62:  
            char_code = ord('_')  
        else:  
            raise ValueError('error')  
        text.append(chr(char_code)) 
    """
    text=[]
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):  
        number = i % 10
        text.append(str(number)) 
             
    return "".join(text)  
   
""" 
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有 
vec = text2vec("F5Sd") 
text = vec2text(vec) 
print(text)  # F5Sd 
vec = text2vec("SFd5") 
text = vec2text(vec) 
print(text)  # SFd5 
"""  
   
# 生成一个训练batch  
def get_next_batch(batch_size=64):  
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])  
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])  
   
    # 有时生成图像大小不是(60, 160, 3)  
    def wrap_gen_captcha_text_and_image():  
        while True:  
            text, image = gen_captcha_text_and_image()  #生成验证码集
            if image.shape == (60, 160, 3):  
                return text, image  
   
    for i in range(batch_size):  
        text, image = wrap_gen_captcha_text_and_image()  
        image = convert2gray(image)  
   
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0  
        batch_y[i,:] = text2vec(text)  
   
    return batch_x, batch_y  
   

   
# 定义CNN  
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])#TensorFlow出入数据的格式为4维，归一化
   
    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #  
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))   
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))   
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))  
    #out_alpha = np.sqrt(2.0/1024)  
   
    # 3 conv layer  
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))  #初始化
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))  #初始化
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv1 = tf.nn.dropout(conv1, keep_prob)  
   
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))  #初始化
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))  #初始化
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2 = tf.nn.dropout(conv2, keep_prob)  
   
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))  #初始化
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))  #初始化
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv3 = tf.nn.dropout(conv3, keep_prob)  
   
    # Fully connected layer  
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))  #初始化，8*20的特征图有64维
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))  #初始化
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  #转换成一维的向量与FC连接
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))  
    dense = tf.nn.dropout(dense, keep_prob)  
   
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))  #初始化
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))  #初始化
    out = tf.add(tf.matmul(dense, w_out), b_out)   
    return out  
   
# 训练  
def train_crack_captcha_cnn():  
    output = crack_captcha_cnn()  
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))  
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    max_idx_p = tf.argmax(predict, 2)  
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
   
        step = 0  
        while True:  
            batch_x, batch_y = get_next_batch(64)  
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})  
            print(step, loss_)  
              
            # 每100 step计算一次准确率  
            if step % 10 == 0:  
                batch_x_test, batch_y_test = get_next_batch(100)  
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})  
                print(step,u'准确率:', acc)  
                # 如果准确率大于80%,保存模型,完成训练  
                if acc > 0.80:  
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)  
                    break  
   
            step += 1  
def crack_captcha(captcha_image):  
    output = crack_captcha_cnn()  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        saver.restore(sess, "./model/crack_capcha.model-1570") 
   
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})  
        text = text_list[0].tolist()  
        return text 
if __name__ == '__main__':
    train = 1
    if train == 0:
        number = ['0','1','2','3','4','5','6','7','8','9']  
        #alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
        #ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        
        text, image = gen_captcha_text_and_image()  #获取4个数字组成的验证码的图片
        print(u"验证码图像channel:", image.shape)  # (60, 160, 3)  
        # 图像大小  
        IMAGE_HEIGHT = 60  
        IMAGE_WIDTH = 160  
        MAX_CAPTCHA = len(text)  #4个
        print(u"验证码文本最长字符数", MAX_CAPTCHA)
        # 文本转向量  
        #char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐  
        char_set = number#数字的识别
        CHAR_SET_LEN = len(char_set)#获取长度10类
        
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])#定义输入的数据格式和规模 
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN]) #None为一个batch，4个*10类
        keep_prob = tf.placeholder(tf.float32) # dropout 
        
        train_crack_captcha_cnn()
    if train == 1:
        number = ['0','1','2','3','4','5','6','7','8','9']  
        IMAGE_HEIGHT = 60  
        IMAGE_WIDTH = 160  
        char_set = number
        CHAR_SET_LEN = len(char_set)#10类
        
        
    
        text, image = gen_captcha_text_and_image()  
        
        
        f = plt.figure()  
        ax = f.add_subplot(111)  
        ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)  
        plt.imshow(image)  
       
        plt.show()  
        
        MAX_CAPTCHA = len(text)
        image = convert2gray(image)  
        image = image.flatten() / 255 #特征尺度的缩放，数据的预处理 
        
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])  
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])  
        keep_prob = tf.placeholder(tf.float32) # dropout 
        
        predict_text = crack_captcha(image)  
        print(u"正确: {}  预测: {}".format(text, predict_text))  
    
    
    
    
