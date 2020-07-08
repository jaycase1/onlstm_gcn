from keras.layers import Layer,Embedding
from on_lstm import ONLSTM
import keras
from keras import backend as K
import tensorflow as tf
maxLen = 216

class GCN_Convlution(Layer):
    def __init__(self,embedding_dim,ndim,num_levels,layers,word_size,**kwargs):
        super(GCN_Convlution,self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.ndim = ndim
        self.num_levels = num_levels
        self.word_size = word_size
        self.layers = layers


    def build(self, input_shape):
        self.embedding = Embedding(self.word_size,self.embedding_dim,input_length=maxLen)

        self.onLstms = [ONLSTM(units=self.ndim,levels=self.num_levels,return_sequences=True,dropconnect=0.3) for i in range(self.layers)]
        self.graph_tran = self.add_weight(name="graph_trans",shape=(self.ndim,self.ndim),initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None))
        self.onLstM_1 = ONLSTM(units=self.ndim,levels=self.num_levels,return_sequences=False)
        self.onLstM_2 = ONLSTM(units=self.ndim, levels=self.num_levels, return_sequences=False)
        super(GCN_Convlution,self).build(input_shape)


    def compute_output_shape(self, input_shape):
        return input_shape[0] + (self.ndim * 2 , )


    def call(self, inputs, **kwargs):
        print("in call")
        assert len(inputs) == 2
        ids = inputs[0]
        print("ids, ",ids)
        graphs = inputs[1]
        print("graphs, ",graphs)
        ids = self.embedding(ids)
        for onlstm in self.onLstms:
            ids = onlstm(ids)
        graphs = tf.matmul(graphs,ids)
        graphs = tf.matmul(graphs,self.graph_tran)
        ids = self.onLstM_1(ids)
        graphs = self.onLstM_1(graphs)
        print(42,K.mean(graphs,1).shape)
        return K.concatenate((ids,graphs),-1)