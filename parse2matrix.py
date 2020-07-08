import sys, os
from pyltp import Segmentor, Postagger, Parser
from scipy import sparse as sp
from nltk import DependencyGraph
import numpy as np

relation_type = {}
class LtpParsing(object):
    def __init__(self, model_dir='E:\ltp_data_v3.4.0\ltp_data_v3.4.0'):
        # 分词
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(model_dir, "cws.model"))
        # 词性标注
        self.postagger = Postagger()
        self.postagger.load(os.path.join(model_dir, "pos.model"))
        # 语义分析
        self.parser = Parser()
        self.parser.load(os.path.join(model_dir, "parser.model"))

    def par(self, instr):
        line = instr.strip()
        # 分词
        words = self.segmentor.segment(line)
        len_ = len(words)
        parse_matrix = np.eye(len_,dtype="int32")
        # self.segmentor.load_with_lexicon('lexicon')  # 使用自定义词典，lexicon外部词典文件路径
        # print('分词：' + '\t'.join(words))

        # 词性标注
        postags = self.postagger.postag(words)
        # print('词性标注：' + '\t'.join(postags))
        # 词性标注得到结果的长度和分词结果一致
        # len(words) == len(postags)
        # 句法分析
        arcs = self.parser.parse(words, postags)
        #print(dir(arcs[0]))
        #for arc in arcs:
            #print(arc.head,arc.relation)

        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            j = arcs[i].head-1
            if arcs[i].relation not in relation_type.keys():
                relation_type[arcs[i].relation] = len(relation_type)
            relation_num = relation_type[arcs[i].relation]
            parse_matrix[i][j] = 1
            parse_matrix[j][i] = 1

        return sp.csc_matrix(parse_matrix)

    def release_model(self):
        # 释放模型
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()

