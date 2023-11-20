import pandas as pd
import requests
import torch.nn as nn
from tqdm import tqdm
import pickle
#################################API格式使用concept graph##############################
def concept_graph_features():
    df = pd.read_csv('./data-concept-instance-relations.txt', sep="\t",names=['Instance','Concept','Relations'])
    print(df[:50])

def concept_grpah_api(entity,topK):
    requests.packages.urllib3.disable_warnings()
    form_header = {"User-Agent": "Chrome/68.0.3440.106"}
    data = {
        'instance':entity,
        'topK':topK
    }
    response = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByProb',params=data,headers=form_header,verify=False)
    #response = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=apple&topK=10&smooth=10',verify=False)
    print(response.json().keys())


#################################本地使用concept graph##############################
def loadingInstance2concept(path='../conceptgraph/instance2concept.pickle'):
    with tqdm(total=1, desc=f'loading Instance2concept file') as pbar:
        with open(path, mode='rb') as f:
            instance2concept = pickle.load(f)
        pbar.update(1)
    return instance2concept

def instance2conept(ins2cpt: dict, instance: str, top=2) -> list:
    '''给定实例，返回其对应的概念，最多两个'''
    concept = ins2cpt.get(instance)
    if concept == None:
        concept = ['unknowConcept1', 'unknowConcept2']
    elif len(concept) == 1:
        concept.append('unknowConcept1')
    else:
        concept = concept[:top]
    return concept[0],concept[1]

######################################MultiHeadAttention##############################
class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        # 注意 Q，K，V的在句子长度这一个维度的数值可以一样，可以不一样。
        # K: [64,10,300], 假设batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], 假设batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], 假设batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10，这里用“0”来指示哪些位置的词向量不能被attention到，比如padding位置，当然也可以用“1”或者其他数字来指示，主要设计下面2行代码的改动。
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 50 和 6 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x


## batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
#query = torch.rand(64, 12, 300)
## batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
#key = torch.rand(64, 10, 300)
## batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
#value = torch.rand(64, 10, 300)
#attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
#output = attention(query, key, value)
### output: torch.Size([64, 12, 300])
#print(output.shape)

########################################实体类型判断#################################
import spacy

def get_ent_type(nlp,ent):
    """输入实体，返回实体的类型"""
    #nlp = spacy.load("en_core_web_lg")
    doc = nlp(str(ent))
    res = [ent.label_ for ent in doc.ents]
    del nlp
    #print(res)
    try:
        assert len(res)==1
        if res[0]=='ORG':
            return 'organization'
        elif res[0]=='GPE':
            return 'location'
        elif res[0]=='PERSON':
            return 'person'
        elif res[0]=='DATE':
            return 'date'
        elif res[0]=='TIME':
            return 'time'
    except Exception as e:
        return 'entity'

###################################################################################
from nltk.corpus import wordnet as wn
wn.synsets('ads')#查询一个词所在的所有词集
wn.synset('apple.n.01').definition()#查询一个同义词集的定义
wn.synset('dog.n.01').examples()#查询词语一个词义的例子

#################################Concpet Net#######################################
def getConcpetNet():
    obj = requests.get('http://api.conceptnet.io/c/en/example').json()
    return obj
   
###################################################################################
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_lg")
sent = "He went to play basketball"
def getDependecyParser(nlp,sent):
    """对句子进行依存句法分析"""
    doc = nlp(sent)
    #print(nlp.pipe_names)#['tagger'，'parser'，'ner']
    nlp.disable_pipes('tagger', 'ner')#禁用除了依存句法的其他功能
    # 依存分析
    for token in doc:
        print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
    #displacy.render(doc, style='dep', jupyter=True, options = {'distance': 90})#可视化

    


###################################################################################
if __name__ == '__main__':
    getDependecyParser(nlp,sent)