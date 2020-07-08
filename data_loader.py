from math import ceil
from data_utils import load_pkl
from scipy import sparse as sp
import numpy as np
from keras.utils import to_categorical

words_id = load_pkl("waimai/dump_data/waimai_corpus.ids")
words_lables = load_pkl("waimai/dump_data/waimai_corpus.labels")
word_graphs = load_pkl("waimai/dump_data/waimai_corpus.graph")
train_split = 0.85
valid_split = 0.1
test_split = 0.05
train_len = int(len(words_id) * train_split)
train_valid_len = int(len(words_id) * (train_split + valid_split))

train_ids = words_id[:train_len]
train_labels = words_lables[:train_len]
train_graphs = word_graphs[:train_len]

valid_ids = words_id[len(train_ids):train_valid_len]
valid_labels = words_lables[len(train_labels):train_valid_len]
valid_graphs = word_graphs[len(train_graphs):train_valid_len]

test_ids = words_id[len(train_ids)+len(valid_ids):]
test_lables = words_lables[len(train_labels)+len(valid_labels):]
test_graphs = word_graphs[len(train_graphs)+len(valid_graphs):]
maxLen = 216
print("train_size:{0},valid_size:{1},test_size:{2}".format(len(train_labels),len(valid_labels),len(test_lables)))





def padding0tosp(spMatrix,maxLen):
    '''
    :param spMatrix: 稀疏矩阵
    :param maxLen:  稀疏矩阵需要填补到的最大长度
    :return:
    '''
    spMatrix = sp.coo_matrix(spMatrix)
    row = spMatrix.row.tolist()
    col = spMatrix.col.tolist()
    data = spMatrix.data.tolist()
    return sp.coo_matrix((data,(row,col)),shape=(maxLen,maxLen)).toarray()


def data_generator(ids,labels,graphs,batch_size=32):
    assert len(ids) == len(labels) and len(ids) == len(graphs)
    steps = ceil(len(ids)/batch_size)
    while True:
        for step in range(steps):
            if step<steps-1:
                temp_ids = ids[step * batch_size : (step+1) * batch_size]
                temp_graphs = graphs[step * batch_size : (step+1) * batch_size]
                temp_labels = labels[step * batch_size : (step+1) * batch_size]
            else:
                temp_ids = ids[step * batch_size : ]
                temp_graphs = graphs[step * batch_size : ]
                temp_labels = labels[step * batch_size : ]
            len_ = max([len(ids) for ids in temp_ids])
            # 对于不定长的序列  长度不够的情况下 在后面补零
            temp_ids = np.array([ids + [0,] * (len_ - len(ids)) for ids in temp_ids])
            temp_graphs = np.array([padding0tosp(graph,len_) for graph in temp_graphs])
            temp_labels = to_categorical(np.array(temp_labels),num_classes=2)
            #temp_labels = temp_labels[:,np.newaxis,:]

            yield ({'input_1': temp_ids, 'input_2': temp_graphs}, {'output': temp_labels})


train_data = data_generator(train_ids,train_labels,train_graphs)
valid_data = data_generator(valid_ids,valid_labels,valid_graphs)
test_data = data_generator(test_ids,test_lables,test_graphs)


