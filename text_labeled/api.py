
# 代码首先引入三个必备的package,分别是os主要用于操作文件路径，
# jieba用于分词处理,fileinput用于从文件读取数据到内存.
import os,jieba,fileinput
from django.http import HttpResponse
# 定义了用户自定义词典路径,和停用词典路径，即在该代码文件路径下应有userdict.txt和stopdict.txt文件
userdict_path = os.path.join(os.path.dirname(__file__),'userdict.txt')
stopdict_path = os.path.join(os.path.dirname(__file__),'stopdict.txt')

# 加载用户自定义词典
jieba.load_userdict(userdict_path)

# 定义输入文本最大长度限制为200
MAX_LIMIT = 200

import json
def handle_cn_text(text: str):
    """用于完成预处理的主要流程, 以原始文本为输入，以分词和去停用词后的词汇列表为输出."""
    # 对输入进行合法性检验
    if not text: return []
    # 使用jieba的cut方法对使用最大限制进行切片的输入文本进行分词
    word_list = jieba.cut(text[:MAX_LIMIT])
    def _load_stop_dict():
        """用于从文件中加载停用词词表"""
        # 使用fileinput加载停用词表至内存,使用字符串的strip()方法去除两端空白符
        # 该方式在window下会报错，而且python3.10以前的版本不支持设置编辑，所以采用其它方式处理
        #stop_word_set  = set(map(lambda  x: x.strip(),fileinput.FileInput(stopdict_path)))
        #return stop_word_set
        stop_word_set = set()
        try:
            with open(stopdict_path, 'r', encoding='utf-8') as file:
                for line in file:
                    stop_word = line.strip()
                    if stop_word:
                        stop_word_set.add(stop_word)
            print("停用词集合中的元素数量:", len(stop_word_set))
            print("部分停用词示例:", list(stop_word_set)[:5])
        except FileNotFoundError:
            print(f"未找到文件: {stopdict_path}")

        return stop_word_set
    # 调用_load_stop_dict()函数
    stop_word_set = _load_stop_dict()




    # 使用高阶函数filter进行循环过滤操作生成最终结果
    word_list = list(filter(lambda x:x not in stop_word_set and len(x) > 1, word_list))
    return word_list

text ="我的眼睛很大很大,可以装得下天空，装得下高山，装得下大海，装得下整个世界；我的眼睛又很小很小，有心事时，就连两行眼泪，也装不下."



from neo4j import GraphDatabase
# 从settings.py配置文件中导入数据库配置NEO4J_CONFIG
#from settings import NEO4J_CONFIG
from django.conf import settings
from config import NEO4J_CONFIG
from itertools import chain

def get_index_map_label(word_list):
    """
        用于获取每个词汇在图谱中对应的类别标签
        该函数以词汇列表为输入, 以词汇出现在词汇列表
        中的索引和对应的[标签, 权重]列表为输出.
        """
    # 对word_list进行合法性检验
    if not word_list: return []
    # 使用GraphDatabase开启一个driver.
    _driver = GraphDatabase.driver(**NEO4J_CONFIG) #解包

    # 开启neo4j的一个session
    with _driver.session() as session:
        def _f(index,word):
            """以词汇列表中一组词索引和词作为输入,
                       返回该索引和词对应的标签列表."""
            # 进行输入的合法性判断
            if not word: return []
            # 建立cypher语句, 它匹配一条图中的路径, 该路径以一个词汇为开端通过一条边连接一个Label节点,
            # 返回标签的title属性,和边的权重, 这正是我们图谱构建时定义的连接模式.
            cypher = "MATCH(a:Vocabulary{name:%r})-[r:Related]-(b:Label)  RETURN b.title, r.weight" % (word)
            record = session.run(cypher)
            #返回结果是：{"标签":权重}
            result = list(map(lambda x: [x[0], x[1]], record))
            print(result)
            if not result: return []
            return [str(index), result]
        #将word_list的索引和词汇作为输入传给_f()函数,并将返回结果做chain操作
        index_map_label = list(chain(*map(lambda x: _f(x[0], x[1]), enumerate(word_list))))
        return index_map_label
def label(text):

    result={"name":"lubincan"}
    print('--------------->', json.dumps(result))
    return result


wordList= ['闪现', '很大', '很大', '装得', '天空', '装得', '高山', '装得', '大海', '装得', '整个', '世界', '眼睛', '很小', '很小', '心事', '两行', '眼泪', '装不下']



# 导入多模型预测函数
from text_labeled.model_train.multithread_predict import request_model_serve

def weight_update(word_list, index_map_label):
    """该函数将分词列表和具有初始概率的标签-概率列表作为输入,将模型预测后的标签-概率列表作为输出"""
    # 首先将列表转化为字典的形式
    # index_map_label >>> ["1", [["美食", 0.735], ["音乐", 0.654]],  "2",  [["美妆", 0.734]] >>>
    # {"1": [["美食", 0.735],["音乐",  0.654]], "2": [["美妆", 0.734]]}
    index_map_label = dict(zip(index_map_label[::2],index_map_label[1::2]))
    for k,v in index_map_label.items():
        #v的长度大于1说明存在歧义现象,大于1说明该词存在两个标签，所以存在歧义
        if len(v) > 1:
            # 获取对应的标签作为参数,即通知服务应该调用哪些模型进行预测.
            label_list = list(map(lambda x:x[0],v))
            # 通过request_model_serve函数获得标签最新的预测概率,并使用字典方式更新.
            # v >>> [["美食": 0.954]]
            v = request_model_serve(word_list,label_list)
            index_map_label.update({k:v})
    # 将字典转化为列表形式
    index_map_label_ =list(chain(*map(lambda x:[x[0],x[1]], index_map_label.items())))
    return index_map_label_

word_list = ['眼睛', '很大', '很大', '装得', '天空', '装得', '高山', '装得', '大海', '装得', '整个', '世界', '眼睛', '很小', '很小', '心事', '两行', '眼泪', '装不下']
index_map_label=["0", [["美妆", 0.654], ["情感故事", 0.765]]]



# 导入可以进行扁平化操作的reduce
# 导入进行合并操作的pandas
from functools import reduce
import pandas as pd
def add(z,y):
    #print('----',z,y)
    #print('---->', z+ y)
    return z+y
def control_increase(index_map_label):
    """以模型预测后的标签-权重列表为输入, 以标签归并后的结果为输出"""
    if not index_map_label: return []
    #  ["2", [["情感故事", 0.765]], "3", [["情感故事",  0.876], ["明星", 0.765]]]
    # 将index_map_label_奇数项即[label, score]取出放在字典中
    # k的数据结构形式:
    # [{'label': '情感故事', 'score': 0.765}, {'label': '情感故事', 'score': 0.876},
    #  {'label': '明星', 'score': 0.765}]
    k = list(map(lambda x:{"label":x[0],"score":x[1]},reduce(lambda z,y: add(z,y),index_map_label[1::2])))
    # 使用pandas中的groupby方法进行合并分值
    df = pd.DataFrame(k)
    df_ = df.groupby(by=['label'])['score'].sum()
    return df_

index_map_label = ["2", [["情感故事", 0.765]], "3", [["情感故事",  0.876], ["明星", 0.765]]]



#实现概率归一化与父标签检索过程
import numpy as np
def father_label_and_normalized(df_):
    """
       以概率调整后的DataFrame对象为输入, 以整个系统的最终结果为输出
       输入样式为:DataFrame<[[“LOL”, 1.465]]>
       输出样式为:[{“label”: “LOL”, “score”: “0.811”, “related”:[“游戏”]}]
       """
    def _sigmoid(x):
        y = 1.0/ (1.0+np.exp(-x))
        return round(y,3)

    def _sg(pair):
        """获得单个标签的父级标签和归一化概率"""
        # 使用GraphDatabase开启一个driver.
        _driver = GraphDatabase.driver(**NEO4J_CONFIG)
        with _driver.session() as session:
            # 通过关系查询获得从该标签节点直到根节点的路径上的其他Label节点的title属性
            cypher = "MATCH(a:Label{title:%r})<-[r:Contain*1..3]-(b:Label) WHERE b.title<>'泛娱乐' RETURN b.title" % pair[0]
            record = session.run(cypher)
            print('-->',record)
            result = list(map(lambda x: x[0], record))
        return {"label": pair[0], "score": _sigmoid(pair[1]), "related": result}
        # 遍历所有的标签

    ## 因为没有构建图谱，所以related匹配不到任何父标签
    #[{'label': '情感故事', 'score': 0.838, 'related': []}, {'label': '明星', 'score': 0.682, 'related': []}]
    return list(map(_sg, df_.to_dict().items()))
if __name__ == "__main__":
    print('去停用词和切词', handle_cn_text(text))
    print('索引与标签映射', get_index_map_label(wordList))
    print('权重更新', weight_update(word_list, index_map_label))
    print('概率调整归一化调整', control_increase(index_map_label))
    print('归一化处理',father_label_and_normalized(control_increase(index_map_label)))

