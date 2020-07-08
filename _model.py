from keras.layers import *
from keras.models import Model
from on_lstm import ONLSTM
from data_loader import train_data,valid_data,test_data
from graph_convolution import GCN_Convlution
import json
import argparse
import tensorflow as tf
from metrics_ import Metrics
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import LSTM
from keras.optimizers import Adam

from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
batch_size = 30
LR = 0.001
EPOCH = 20
word_embedding_dim = 300
lstm_dim = 300
num_levels = 15
lstm_layers = 2
maxLen = 216

class GCN(Layer):
    def __init__(self):
        super(GCN,self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[0][-1]),int(input_shape[0][-1])),initializer='uniform',trainable=True)
        super(GCN,self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        ids = inputs[0]
        graphs = inputs[1]
        out_1 = tf.matmul(tf.matmul(graphs,ids),self.W)
        return ids + out_1

    def compute_output_shape(self, input_shape):
        return input_shape[0]


parser =  argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='weibo')

with open("dict.json",'r',encoding='UTF-8') as f:
    word2id = json.load(f)
    f.close()

inputs = Input(shape=(None,),name="input_1",dtype="int32")
ids = Embedding(input_dim=len(word2id),output_dim=128)(inputs)
ids = ONLSTM(128,16,return_sequences=True)(ids)
ids = ONLSTM(128,16,return_sequences=True)(ids)
graphs = Input(shape=(None,None),name="input_2",dtype="float32")
conv1 = GCN()([ids,graphs])

conv2 = ONLSTM(128,8)(conv1)
conv2 = Dropout(0.5)(conv2)
y_ = Dense(32)(conv2)
y = Dense(2,activation="softmax",name="output")(y_)
model = Model([inputs,graphs],y)
model.summary()
metric = Metrics(valid_data,50,test_data,25)
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["binary_accuracy"])
model.fit_generator(train_data,
                    callbacks=[TensorBoard(log_dir=".log"),metric],
                    steps_per_epoch=425,
                    validation_data=valid_data,
                    validation_steps=50,
                    epochs=EPOCH)









