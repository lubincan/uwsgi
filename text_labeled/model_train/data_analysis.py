import pandas as pd
import jieba
import os
user_path=os.path.dirname(__file__)

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
def get_data_labels(csv_path):
    """获得训练数据和对应的标签, 以正负样本的csv文件路径为参数"""
    # 使用pandas读取csv文件至内存
    df = pd.read_csv(os.path.join(user_path,csv_path),header=None,sep='\t')
    print(df.head(10))
    print(df.tail(10))
    # 对句子进行分词处理并过滤掉长度为1的词
    train_data = list(map(lambda x:list(filter(lambda x: len(x)>1, jieba.cut(x))),df[1].values))
    # 取第0列的值作为训练标签
    train_labels = df[0].values
    print(csv_path)
    return train_data,train_labels


def get_data_labels_():
    csv_paths = ["./movie/sample.csv","./star/sample.csv","./fashion/sample.csv","./beauty/sample.csv"]
    return list(map(get_data_labels,csv_paths))


import os
from collections import Counter

def pic_show(pic, pic_path, pic_name):
    """用于图片显示，以图片对象和预保存的路径为参数"""
    if not os.path.exists(os.path.join(user_path,pic_path)):os.mkdir(pic_path)
    pic.savefig(os.path.join(user_path,pic_path, pic_name))

def get_labels_distribution(train_labels, pic_path, pic_name="ld.png"):
    """获取正负样本数量的基本分布情况"""
    # class_dict >>> {1: 3995, 0: 4418}
    class_dict = dict(Counter(train_labels))
    print(class_dict)
    df = pd.DataFrame(list(class_dict.values()), list(class_dict.keys()))
    # 设置为非交互模式
    plt.switch_backend('Agg')
    pic = df.plot(kind='bar', title="type sample").get_figure()
    pic_show(pic, pic_path, pic_name)
    return class_dict,pic_path

def get_labels_distribution_():
    list_ = []
    # 图片的存储路径
    pic_path = "./movie/"
    # 图片的名字, 默认是ld.png
    pic_name = "ld.png"
    list_.append(get_labels_distribution(get_data_labels("./movie/sample.csv")[1], pic_path, pic_name))

    pic_path = "./star/"
    # 图片的名字, 默认是ld.png
    pic_name = "ld.png"
    list_.append(get_labels_distribution(get_data_labels("./star/sample.csv")[1], pic_path, pic_name))

    pic_path = "./beauty/"
    # 图片的名字, 默认是ld.png
    pic_name = "ld.png"
    list_.append(get_labels_distribution(get_data_labels("./beauty/sample.csv")[1], pic_path, pic_name))

    pic_path = "./fashion/"
    # 图片的名字, 默认是ld.png
    pic_name = "ld.png"
    list_.append(get_labels_distribution(get_data_labels("./fashion/sample.csv")[1], pic_path, pic_name))

    '''
    样本分布统计情况如下
    ./movie/sample.csv
    {1: 43945, 0: 66495}
    ./star/sample.csv
    {1: 7161, 0: 66495}
    ./beauty/sample.csv
    {1: 26037, 0: 66495}
    ./fashion/sample.csv
    {1: 15389, 0: 66495}
    '''

    return list_

#获取句子长度分布过程的代码分析:
def get_sentence_length_distribution(train_data, pic_path, pic_name="sld.png"):
    """该函数用于获得句子长度分布情况"""
    sentence_len_list = list(map(len,train_data))
    # len_dict >>> {38: 62, 58: 18, 40: 64, 35: 83,....}
    len_dic = dict(Counter(sentence_len_list))
    len_list = list(zip(len_dic.keys(),len_dic.values()))
    # len_list >>> [(1, 3), (2, 20), (3, 51), (4, 96), (5, 121), (6, 173), ...]
    len_list.sort(key=(lambda x: x[0]))
    df = pd.DataFrame(list(map(lambda x:x[1],len_list)),list(map(lambda x: x[0],len_list)))
    # 设置为非交互模式
    plt.switch_backend('Agg')
    ax = df.plot(kind='bar', figsize=(18, 18), title="句子长度分布图")
    ax.set_xlabel("句子长度")
    ax.set_ylabel("该长度出现的次数")
    pic = ax.get_figure()
    pic_show(pic, pic_path, pic_name)
    print(pic_path)
    return len_list,pic_path


def get_sentence_length_distribution_():
    list_=[]
    pic_path = "./movie/"
    # 图片的名字, 默认是sld.png
    pic_name = "sld.png"
    list_.append(get_sentence_length_distribution(get_data_labels("./movie/sample.csv")[0], pic_path, pic_name))

    pic_path = "./star/"
    # 图片的名字, 默认是sld.png
    pic_name = "sld.png"
    list_.append(get_sentence_length_distribution(get_data_labels("./star/sample.csv")[0], pic_path, pic_name))

    pic_path = "./fashion/"
    # 图片的名字, 默认是sld.png
    pic_name = "sld.png"
    list_.append(get_sentence_length_distribution(get_data_labels("./fashion/sample.csv")[0], pic_path, pic_name))

    pic_path = "./beauty/"
    # 图片的名字, 默认是sld.png
    pic_name = "sld.png"
    list_.append(get_sentence_length_distribution(get_data_labels("./beauty/sample.csv")[0], pic_path, pic_name))
    return list_


from itertools import chain
#获取常见词频分布的代码分析过程:,词汇总数，词频分布
def get_word_frequency_distribution(train_data, pic_path, pic_name="wfd.png"):
    # 设置为非交互模式
    plt.switch_backend('Agg')
    """该函数用于获得词频分布"""
    vocab_size = len(set(chain(*train_data)))
    print("所有样本共包含不同词汇数量为：", vocab_size)
    # 获取常见词分布字典，以便进行绘图
    # common_word_dict >>> {'电影': 1548, '自己': 968, '一个': 850, '导演': 757, '现场': 744, ...}
    common_word_dict = dict(Counter(chain(*train_data)).most_common(50)) #用于指定返回出现次数最多的前 n 个元素及其计数
    df = pd.DataFrame(list(common_word_dict.values()),
                       list(common_word_dict.keys()))
    pic = df.plot(kind='bar', figsize=(18, 18), title="常见词分布图").get_figure()
    pic_show(pic, pic_path, pic_name)
    return common_word_dict,pic_path


def get_word_frequency_distribution_():
    list_=[]
    pic_path = "./movie/"
    # 图片的名字, 默认是wfd.png
    pic_name = "wfd.png"
    list_.append(get_word_frequency_distribution(get_data_labels("./movie/sample.csv")[0], pic_path, pic_name))

    pic_path = "./star/"
    # 图片的名字, 默认是wfd.png
    pic_name = "wfd.png"
    list_.append(get_word_frequency_distribution(get_data_labels("./star/sample.csv")[0], pic_path, pic_name))

    pic_path = "./fashion/"
    # 图片的名字, 默认是wfd.png
    pic_name = "wfd.png"
    list_.append(get_word_frequency_distribution(get_data_labels("./fashion/sample.csv")[0], pic_path, pic_name))

    pic_path = "./beauty/"
    # 图片的名字, 默认是wfd.png
    pic_name = "wfd.png"
    list_.append(get_word_frequency_distribution(get_data_labels("./beauty/sample.csv")[0], pic_path, pic_name))
    return list_

if __name__=='__main__':

    #样本csv文件路径
    # get_data_labels函数得到的train_labels
    # print(get_data_labels_()) #标签
    # 通过get_data_labels得到的train_data(需要进行均衡切片)
    # get_labels_distribution_();#正负样本分布
    # 图片的存储路径
    #get_sentence_length_distribution_ #句子长度分布
    # 通过get_data_labels得到的train_data(需要进行均衡切片)
    get_word_frequency_distribution_()#词汇总数，词频分布


