from keras import backend as K
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from _model import GCN
from on_lstm import ONLSTM
from keras.models import load_model


def gen2result(model,gen,steps):
    val_predict = []
    val_targ = []
    for i in range(steps):
        x,y = gen.__next__()
        y = np.argmax(y['output'],-1)
        predict = np.argmax(model.predict([x['input_1'],x['input_2']]),-1)
        val_predict.extend(predict.tolist())
        val_targ.extend(y.tolist())
    return val_predict,val_targ





class Metrics(Callback):
    def __init__(self,validation_data,valid_steps,test_data,test_steps):
        self.validation_data = validation_data
        self.test_data = test_data
        self.valid_steps = valid_steps
        self.test_steps = test_steps


    def on_train_begin(self, logs={}):
        self._best_precisions = -100.
        self._best_f1 = -100.
        self._best_recall = -100.
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict,val_targ = gen2result(self.model,self.validation_data,self.valid_steps)
        #val_predict = (numpy.asarray(self.model.predict(
            #self.validation_data[0]))).round()
        #val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        print("precision:{0:.4f},recall:{1:.4f},f1:{2:.4f}".format(_val_precision,_val_recall,_val_f1))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        if (self._best_precisions<_val_precision):
            self.epoch_best = epoch
            self._best_precisions = _val_precision
            self._best_f1 = _val_f1
            self._best_recall = _val_recall
            self.model.save("model_{0}_epoch.h5".format(epoch))
        return


    def on_train_end(self, logs=None):
        print("the best model's precision:{0:.4f},recall:{1:.4f},f1:{2:.4f}".format(self._best_precisions,self._best_recall,self._best_f1))
        test_predict,test_targ = gen2result(self.model,self.test_data,self.test_steps)
        _val_f1 = f1_score(test_targ, test_predict)
        _val_recall = recall_score(test_targ, test_predict)
        _val_precision = precision_score(test_targ, test_predict)
        print("in test data precision:{0:.4f},recall:{1:.4f},f1:{2:.4f}".format(_val_precision, _val_recall, _val_f1))
        best_model = load_model("model_{0}_epoch.h5".format(self.epoch_best),custom_objects={"GCN":GCN,"ONLSTM":ONLSTM})
        _predict,_targ = gen2result(best_model,self.test_data,self.test_steps)
        _val_f1 = f1_score(_targ, _predict)
        _val_recall = recall_score(_targ, _predict)
        _val_precision = precision_score(_targ, _predict)
        print("the saved model in test data precision:{0:.4f},recall:{1:.4f},f1:{2:.4f}".format(_val_precision, _val_recall, _val_f1))
