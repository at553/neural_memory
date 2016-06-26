import tensorflow as tf, numpy as np
import tflearn, random, os
from PIL import Image

#image data creation
im = Image.open('/Users/avinashthangali/PycharmProjects/compression/lasp-logo.grayscale.art-only.small_.png').convert('L')
imarr = np.array(im)
img = Image.fromarray(imarr)
img.show()
inp = []
lab = []
xrange, yrange = imarr.shape

#format training data
for x in range(1, xrange-1):
    for y in range(1, yrange-1):
        entry = [imarr[x-1][y], imarr[x-1][y-1], imarr[x-1][y+1], imarr[x][y-1]]
        inp.append(entry)
        lab.append(imarr[x][y])

lab = np.array(lab, dtype='int32')


#convert labels to one_hot dense representation
one_hot = lab
lab = tf.one_hot(lab, 256, on_value=10)

#initialize tensorflow session
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
lab = np.array(sess.run(lab))
inp = np.array(inp)

#input layer for both original and prediction models
input_layer = tflearn.input_data(shape=[None, 4])


#define initial DNN model
dense1 = tflearn.fully_connected(input_layer, 16, activation='relu', regularizer='L2', weight_decay=0.001)
dense2 = tflearn.fully_connected(dense1, 64, activation='relu', regularizer='L2', weight_decay=0.001)
softmax = tflearn.fully_connected(dense2, 256, activation='softmax')

#define accuracy metrics and train model
top_1 = tflearn.metrics.Top_k(1)
net = tflearn.regression(softmax, optimizer='adam', metric=top_1, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(inp, lab, n_epoch=1000, validation_set=(inp, lab), show_metric=True, run_id="dense_model")



#retrieve weights and biases of trained DNN
softmaxW = (model.get_weights(softmax.W))
softmaxB = model.get_weights(softmax.b)
dense1W = model.get_weights(dense1.W)
dense1B = model.get_weights(dense1.b)
dense2B = model.get_weights(dense2.b)
dense2W = model.get_weights(dense2.W)

#define predictor model
dense1p = tflearn.fully_connected(input_layer, 16, activation='relu', regularizer='L2', weight_decay=0.001)
dense2p = tflearn.fully_connected(dense1p, 64, activation='relu', regularizer='L2', weight_decay=0.001)
softmaxp = tflearn.fully_connected(dense2p, 256, activation='softmax')

#create predictor model
top_1 = tflearn.metrics.Top_k(1)
net = tflearn.regression(softmaxp, optimizer='adam', metric=top_1, loss='categorical_crossentropy')
model2 = tflearn.DNN(net, tensorboard_verbose=0)

#set weights to those obtained by trained model
model2.set_weights(dense1p.W, dense1W)
model2.set_weights(dense1p.b, dense1B)
model2.set_weights(dense2p.W, dense2W)
model2.set_weights(dense2p.b, dense2B)
model2.set_weights(softmaxp.W, softmaxW)
model2.set_weights(softmaxp.b, softmaxB)


lar = []

#create image array from trained model
dicti = dict()
for x in range(1, xrange-1):
    elo = []
    for y in range(1, yrange-1):
        #elo.append(imarr[x][y])

        try:
            e1 = dicti[x-1, y]
        except KeyError:
            e1 = imarr[x-1, y]
        try:
            e2 = dicti[x - 1, y-1]
        except KeyError:
            e2 = imarr[x-1, y-1]
        try:
            e3 = dicti[x - 1, y+1]
        except KeyError:
            e3 = imarr[x-1, y+1]
        try:
            e4 = dicti[x, y-1]
        except KeyError:
            e4 = imarr[x, y-1]

        entry = [e1, e2, e3, e4]
        elo.append(np.argmax(model2.predict(np.array(entry).reshape(1, 4))))
        dicti[x, y] = np.argmax(model2.predict(np.array(entry).reshape(1, 4)))
    lar.append(elo)

#create and display resulting image
imgarr = (np.array(lar, dtype='int32'))
imgfin = Image.fromarray(imgarr)
imgfin.convert(mode='L').show()















