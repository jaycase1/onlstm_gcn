import json
import os
from data_utils import create_dict, get_clean_name,get_corpus,txt2csv
dump_dir = "waimai/dump_data/"
if not os.path.exists(dump_dir):
    # 数据预处理 以及 graph保存的地址
    os.mkdir(dump_dir)

txt2csv("waimai/waimai_pos.txt","waimai/waimai_neg.txt")
create_dict(["waimai/waimai.csv"])

with open("dict.json",'r',encoding='UTF-8') as f:
    word2id = json.load(f)
    f.close()

id2word = {}
for key,value in word2id.items():
    id2word[value] = key
ds_corpus_contents = ["waimai_corpus.ids"]
ds_corpus_labels = ["waimai_corpus.labels"]
ds_corpus_graphs = ["waimai_corpus.graph"]
new_filenames = [get_clean_name("waimai/waimai.csv")]

get_corpus(new_filenames,word2id,ds_corpus_contents,ds_corpus_labels,ds_corpus_graphs,dump_dir)