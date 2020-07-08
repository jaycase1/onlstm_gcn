from keras.layers import *
import json
from keras.models import Model
from keras.optimizers import RMSprop
from data_loader import train_data,valid_data
from on_lstm import ONLSTM
import tensorflow as tf
import keras




with open("dict.json",'r',encoding='UTF-8') as f:
    word2id = json.load(f)
    f.close()

EPOCH = 20

max_words = len(word2id)
max_len = 216

inputs = Input(shape=(None,),dtype="int32")
#graphs = Input(shape=(None,None))
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1,128)(inputs)
print(layer)
layer = ONLSTM(128,16,return_sequences=True)(layer)
layer = ONLSTM(128,16)(layer)
layer = Dense(32,activation="relu",name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=RMSprop(),metrics=["accuracy"])

model.fit_generator(train_data,
                    steps_per_epoch=160,
                    validation_data=valid_data,
                    validation_steps=20,
                    epochs=EPOCH)
