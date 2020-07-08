import re
from pyltp import Segmentor
import csv
import json
import os
import pandas as pd
import sys
import numpy as np
import pickle as pkl
from scipy import sparse as sp
from parse2matrix import LtpParsing
np.random.seed(47)
model_dir = 'E:\ltp_data_v3.4.0\ltp_data_v3.4.0'

def normalize_adj(graph):
    """Symmetrically normalize adjacency matrix.
        the graph have already add self-loop
        the output is the graph's Laplacian Operator
    """
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(graph.sum(1)) #D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).todense()
    return graph.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def write_pkl(content,filename,dump_dir):
    filename = dump_dir + filename
    with open(filename,'wb') as f:
        pkl.dump(content,f)


def load_pkl(filename):
    with open(filename,'rb') as f:
        return pkl.load(f)




def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    text = re.sub(r'\[+', "", text)
    text = re.sub(r'\]+', "", text)
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = text.replace("转发微博", "")       # 去除无意义的词语
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    return text.strip()

def read_file_from_csv(filename):
    """
    :param filename: the format must be csv
    :return:
    """
    format = filename.split('.')[-1]
    if format== "csv":
        data = pd.read_csv(filename, encoding='UTF-8')
    elif format == "tsv":
        data = pd.read_table(filename,encoding='UTF-8')
    else:
        Warning("More format haven't develop, also you can overload it")
        sys.exit()

    return data


def get_clean_name(filename,insert_="_clean."):
    names = filename.split('.')
    new_filename = ""
    for name in names[:-1]:
        new_filename += name
    new_filename += insert_
    new_filename += names[-1]
    return new_filename


def create_new_csv(filename,headers,*args):
    names = filename.split(".")
    if names[-1] != "csv":
        names[-1] = ".csv"
        filename = ""
        for name in names:
            filename += name

    assert len(headers) == len(args)
    with open(filename,'w',newline="",encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        rows = zip(*args)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
        csvfile.close()







def create_dict(filenames,dict_path="dict.json"):
    '''
    :param filename:  要读取的corpus列表 要求为csv或者tsv格式
    :param dict_path: 字典要存储的位置
    :return:
    '''
    dict_ = {}
    segmentor = Segmentor()
    segmentor.load(os.path.join(model_dir, "cws.model"))
    for filename in filenames:
        new_filenname = get_clean_name(filename)
        if os.path.exists(new_filenname):
            # 假定新文件被写入的时候  dict_ 文件同时被写入
            print("clean csv have already exist")
        else:
            data = read_file_from_csv(filename)
            len_ = len(data)
            indices = np.arange(len_)
            np.random.shuffle(indices)

            headers = data.columns.values
            labels = data[headers[0]].reindex(indices)
            reviews = data[headers[1]].reindex(indices)
            labels = labels.tolist()
            reviews = reviews.tolist()
            for i, review in enumerate(reviews):
                review = clean(review)
                words = segmentor.segment(review)
                reviews[i] = review
                for word in words:
                    if word not in dict_.keys():
                        dict_[word] = len(dict_)
            create_new_csv(new_filenname, headers, labels, reviews)
            json_dict = json.dumps(dict_)
            with open(dict_path, 'w', encoding="UTF-8") as f:
                f.write(json_dict)
            f.close()
    segmentor.release()



def get_corpus(filenames,word2id,content_filenames,label_filenames,graph_filenames,dump_dir=""):
    lp = LtpParsing()
    segmentor = lp.segmentor
    segmentor.load(os.path.join(model_dir, "cws.model"))
    for index ,filename in enumerate(filenames):
        data = read_file_from_csv(filename)
        labels = data['label']
        contents = data['review']
        assert len(data) == len(labels)
        sentence_ids = []
        sentence_labels = []
        sentence_graphs = []
        for i, content in enumerate(contents):
            if (type(content) != str or content == ""):
                continue
            else:
                words = segmentor.segment(content)
                ids = [word2id[word] for word in words]
                sentence_ids.append(ids)
                sentence_labels.append(labels[i])
                sentence_graphs.append(normalize_adj(lp.par(content)))
        write_pkl(sentence_ids, content_filenames[index], dump_dir)
        write_pkl(sentence_labels, label_filenames[index], dump_dir)
        write_pkl(sentence_graphs, graph_filenames[index], dump_dir)
    lp.release_model()

def read_txt(filename):
    contents = []
    with open(filename,'r',encoding="UTF-8") as f:
        for line in f:
            contents.append(line.strip())
    f.close()
    return contents

def txt2csv(pos_txt,neg_txt):
    '''
    :param pos_txt: 正向文本
    :param neg_txt: 负向文本
    :return:
    '''
    names = pos_txt.split("_")
    filename = names[0] +".csv"
    print(filename)
    headers = ["label","review"]
    pos = read_txt(pos_txt)
    neg = read_txt(neg_txt)
    labels = [1,] * len(pos) + [0,] * len(neg)
    reviews = pos + neg
    create_new_csv(filename,headers,labels,reviews)