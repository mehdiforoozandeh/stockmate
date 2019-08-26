import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
import datetime
from google.colab import drive
drive.mount('/content/gdrive')

tf.reset_default_graph()
path=  '/content/gdrive/My Drive/portfo/'

DATA = path + raw_input("what's your file's name?    ")


st = datetime.datetime.now()
# Parameters
init_learning_rate = 0.00028 # Default= 0.00291
learning_rate_decay = 0.999999 # Default= 0.99991
max_epochs =5000
init_epoch=1
keep_prob = 1
num_layers = 1 # Default=1
perc = 99  # Default=80
n_inputs = 3
D=6
lstm_size = 512  # Default=256
displaystep = max_epochs/100

df = pd.read_csv(DATA)
df = df.drop(["<TICKER>", "<VALUE>", "<OPENINT>", '<PER>'], axis=1)

#print(df.shape)

# df= df.sort_values(by='<DTYYYYMMDD>', ascending=True)

# print df.head(10)

# print df.tail(5)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Close', 'Date', 'First', 'High', 'Low', 'Vol', 'Open', 'Last'])
for i in range(0, len(new_data)):
    new_data['Date'][i] = str(df['<DTYYYYMMDD>'][i]) # <FIRST>  <HIGH>   <LOW> <VOL>  <OPEN><LAST>
    new_data['Close'][i] = df['<CLOSE>'][i]
    new_data['First'][i] = df['<FIRST>'][i]
    new_data['High'][i] = df['<HIGH>'][i]
    new_data['Low'][i] = df['<LOW>'][i]
    new_data['Vol'][i] = df['<VOL>'][i]
    new_data['Open'][i] = df['<OPEN>'][i]
    new_data['Last'][i] = df['<LAST>'][i]
new_data = new_data.sort_values(by='Date', ascending=True)

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
new_data.drop('Vol', axis=1, inplace=True)
print new_data.head(10)
print new_data.shape
# print new_data.tail(5)
# new_data.index = new_data.Date
# print new_data


close_price = new_data

#print close_price.head(10)

lastfeed = np.array(close_price[-n_inputs:], dtype=int)
lastfeed = np.reshape(lastfeed, [-1, n_inputs, D])
print (lastfeed)

def trainsetsplit(cp, percentage):
    ttpoint = int(len(cp) * int(percentage) / 100)
    train = cp[:ttpoint]
    test = cp[ttpoint:]
    return train, test


# ttpoint = int(len(close_price) * int(perc) / 100)

# close_price = close_price.values.reshape(-1, 1)

train, test = trainsetsplit(close_price, perc)

# print train.shape, test.shape

#print(train.shape), (test.shape)
xbatch, ybatch = [], []
for t in range(n_inputs, len(train)):
    xbatch.append(np.array(train[t - n_inputs:t]))
    ybatch.append(np.array(train[t:t+1]))
xbatch = np.reshape(np.array(xbatch), [-1, n_inputs, D])
ybatch = np.reshape(np.array(ybatch), [-1, D])
print xbatch.shape, ybatch.shape

# def Test_next_batch(batchsize):

xbatchtest, ybatchtest = [], []
for t in range(n_inputs, len(test)):
    xbatchtest.append(np.array(test[t - n_inputs:t]))
    ybatchtest.append(np.array(test[t:t+1]))
xbatchtest = np.reshape(np.array(xbatchtest), [-1, n_inputs, D])
ybatchtest = np.reshape(np.array(ybatchtest), [-1, D])
print xbatchtest.shape, ybatchtest.shape

# DEFINE GRAPH

X = tf.placeholder("float32", [None, n_inputs, D])
Y = tf.placeholder("float32", [None, D])
learning_rate = tf.placeholder(tf.float32, None)

weights = tf.get_variable('weight', initializer=tf.random_normal([lstm_size, D]))
biases = tf.get_variable("biases", initializer=tf.random_normal([D]))


def _create_one_cell():
    return tf.contrib.rnn.LSTMCell((lstm_size), activation=tf.nn.elu)
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

def LSTM(x, weights, biases):
    # x = tf.unstack(x, n_inputs, 1)
    # lstm_cell = rnn.BasicLSTMCell(n_hidden, reuse=tf.AUTO_REUSE)
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # return tf.matmul(outputs[-1], weights) + biases

    cell = tf.contrib.rnn.MultiRNNCell(
        [_create_one_cell() for _ in range(num_layers)]
    ) if num_layers > 1 else _create_one_cell()
    val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
    return tf.matmul(last, weights) + biases


logits = LSTM(X, weights, biases)
# pred = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.square(logits - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # AdamOptimizer-RMSPropOptimizer

learning_rates_to_use = [
    init_learning_rate * (
            learning_rate_decay ** max(float(i + 1 - init_epoch), 0.0)
    ) for i in range(max_epochs)]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(max_epochs):
        current_lr = learning_rates_to_use[i]
        sess.run(optimizer, feed_dict={X: xbatch, Y: ybatch, learning_rate: current_lr})
        # train_preds.append(sess.run(logits, feed_dict={X: xbatch}))
        loss = sess.run(cost, feed_dict={X: xbatch, Y: ybatch})

        if i % displaystep == 0:
            print ('Epoch:  ' + str(i) + '/' + str(max_epochs) + '   ===> Loss:' + str(loss))
    print 'Training Done! \n'
    testloss= sess.run(cost, feed_dict={X: xbatchtest, Y: ybatchtest})
    print 'Testing Loss:    '+ str(testloss) + "\n"

    test_preds = sess.run(logits, feed_dict={X: xbatchtest})
    tomorrow = sess.run(logits, feed_dict={X: lastfeed})
    print ("Predictions for tomorrow:  \n" + 'Close:    '+ str(tomorrow[0,0])+"\n"+ 'First:    '+ str(tomorrow[0,1])+"\n"
           + 'High:    ' + str(tomorrow[0,2]) + "\n"+ 'Low:    '+ str(tomorrow[0,3])+"\n"+ 'Open:    '+ str(tomorrow[0,4])+"\n"+
           'Last:    '+ str(tomorrow[0,5])+"\n")



    print '% Accuracy: '+ str((1 - (testloss/(np.mean(ybatchtest))**2)**(0.5))*100)

print "Runtime: " + str(datetime.datetime.now() - st)
