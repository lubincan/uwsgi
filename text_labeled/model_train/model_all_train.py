import sys
import os
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)
# 导入用于对象保存与加载的joblib
#from sklearn.externals import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 导入从样本csv到内存的get_data_labels函数
from data_analysis import get_data_labels
import joblib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

user_path=os.path.dirname(__file__)
#进行词汇映射的代码分析过程:


from text_labeled import api

from sklearn.model_selection import train_test_split

def word_map(csv_path, tokenizer_path, cut_num):
    """进行词汇映射，以训练数据的csv路径和映射器存储路径以及截断数为参数"""
    # 使用get_data_labels函数获取简单处理后的训练数据和标签
    train_data,train_labels = get_data_labels(csv_path)
    # 进行正负样本均衡切割, 使其数量比例为1:1
    #train_data = train_data[:-cut_num]
    #train_labels = train_labels[:-cut_num]
    train_data = train_data[:cut_num]
    train_labels = train_labels[:cut_num]
    # 实例化一个词汇映射器对象
    '''
    功能：该参数是一个布尔值，用于指定是按字符级别还是单词级别进行分词。
    若 char_level 为 False（默认值），则按单词级别进行分词。也就是将文本拆分成一个个单词，然后为每个单词分配一个唯一的整数索引。
    若 char_level 为 True，则按字符级别进行分词。即把文本拆分成一个个字符，为每个字符分配一个唯一的整数索引
    '''
    t = Tokenizer(num_words=None,char_level=False) #nb_words保留场景最高词频数量，char_level
    # 使用映射器拟合现有文本数据
    t.fit_on_texts(train_data)
    # 使用joblib工具保存映射器
    joblib.dump(t,tokenizer_path)
    # 使用映射器转化现有文本数据
    x_train = t.texts_to_sequences(train_data)
    # 获得标签数据
    y_train = train_labels
    #print('x_train--->',x_train.shape(),'y_tain-->',y_train.shape())
    return x_train, y_train


def word_map_():

    # 对应的样本csv路径
    csv_path = "./movie/sample.csv"
    # 词汇映射器保存的路径
    tokenizer_path = "./movie/Tokenizer"
    # 截断数
    cut_num = 42000
    print(word_map(csv_path, tokenizer_path, cut_num))

    # 对应的样本csv路径
    csv_path = "./star/sample.csv"
    # 词汇映射器保存的路径
    tokenizer_path = "./star/Tokenizer"
    # 截断数
    cut_num = 8000
    print(word_map(csv_path, tokenizer_path, cut_num))

    # 对应的样本csv路径
    csv_path = "./fashion/sample.csv"
    # 词汇映射器保存的路径
    tokenizer_path = "./fashion/Tokenizer"
    # 截断数
    cut_num = 15000
    print(word_map(csv_path, tokenizer_path, cut_num))

    # 对应的样本csv路径
    csv_path = "./beauty/sample.csv"
    # 词汇映射器保存的路径
    tokenizer_path = "./beauty/Tokenizer"
    # 截断数
    cut_num = 2525
    print(word_map(csv_path, tokenizer_path, cut_num))


#向量截断对齐的代码分析过程:
#from keras.preprocessing import sequence
from tensorflow.keras.preprocessing import sequence
# cutlen根据数据分析中句子长度分布，覆盖90%语料的最短长度.
cutlen = 60
def padding(x_train, cutlen):
    return sequence.pad_sequences(x_train, cutlen)


#加入n-gram特征过程的代码分析:
# 根据样本集最大词汇数选择最大特征数，应大于样本集最大词汇数
max_features = 25000
# n-gram特征的范围，一般选择为2
ngram_range = 2
def create_ngram_set(input_list, ngram_value=2):
    """
       从列表中提取n-gram特征
       >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
       {(4, 9), (4, 1), (1, 4), (9, 4)}
       """
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))




import numpy as np

#增加ngram后获取新的最大特性
def get_ti_and_nmf(x_train, ti_path, ngram_range):
    """new max features简写nmf,从训练数据中获得token_indice和新的max_features"""
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # 创建一个盛装n-gram特征的集合.
    ngram_set = set()
    # 遍历每一个数值映射后的列表
    for input_list in x_train:
        # 遍历可能存在2-gram, 3-gram等
        for i in range(2,ngram_range+1):
             # 获得对应的n-gram表示，range(2,ngram_range+1)是为了给底下的ngram_value赋值
             set_of_ngram = create_ngram_set(input_list,ngram_value=i)
             # 更新n-gram集合
             ngram_set.update(set_of_ngram)

        # 去除掉(0, 0)这个2-gram特征
        ngram_set.discard(tuple([0]*ngram_range))
        # 将n-gram特征映射成整数.
        # 为了避免和之前的词汇特征冲突，n-gram产生的特征将从max_features+1开始
        start_index=max_features+1
        # 得到对n-gram表示与对应特征值的字典
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        # 将token_indice写入文件以便预测时使用
        with open(ti_path, "w",encoding='utf-8') as f:
            f.write(str(token_indice))
        # token_indice的反转字典，为了求解新的最大特征数
        indice_token = {token_indice[k]: k for k in token_indice}
        # 获得加入n-gram之后的最大特征数
        new_max_features = np.max(list(indice_token.keys())) + 1
        return token_indice, new_max_features

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    sequences 是x-train,  token_indice是最大特征
    将n-gram特征加入到训练数据中
    如: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    """
    new_sequences = []
    # 遍历序列列表中的每一个元素作为input_list, 即代表一个句子的列表
    for input_list in sequences:
        # copy一个new_list
        new_list = np.array(input_list[:]).tolist()
        # 遍历n-gram的value，至少从2开始
        for ngram_value in range(2, ngram_range + 1):
            # 遍历各个可能的n-gram长度
            for i in range(len(new_list) - ngram_value + 1):
                # 获得input_list中的n-gram表示
                ngram = tuple(new_list[i:i + ngram_value]) #采用元组是不可变对象，比较安全
                # 如果在token_indice中，则追加相应的数值特征
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences #np.array(new_sequences)

#将向量进行最长补齐过程,因为增加ngram特性后每个句子长度发生了变化，所以需要再次补齐
def align(x_train):
    """用于向量按照最长长度进行补齐"""
    # 获得所有句子长度的最大值
    maxlen = max(list(map(lambda x:len(x),x_train)))
    # 调用padding函数
    x_train = padding(x_train,maxlen)
    return x_train,maxlen

# 首先导入keras构建模型的必备工具包
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D

# 定义词嵌入维度为50
embedding_dims = 50
def model_build(maxlen, new_max_features):
    """该函数用于模型结构构建"""

    # 在函数中，首先初始化一个序列模型对象
    model = Sequential()
    # 然后首层使用Embedding层进行词向量映射
    model.add(Embedding(new_max_features,embedding_dims,input_length=maxlen))
    # 然后用构建全局平均池化层，减少模型参数，防止过拟合,sigmoid层来进行分类
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1,activation='sigmoid'))
    return model

def model_build_():
    # 最大对齐长度, 即输入矩阵中每条向量的长度
    maxlen = 119

    # 最大特征数, 即输入矩阵中元素的最大值
    new_max_features = 143307

    # 词嵌入的数量, 使用50维
    embedding_dims = 50
    model_build(maxlen,new_max_features)

import tensorflow as tf
def model_compile(model:Sequential):
    """用于选取模型的损失函数和优化方法"""
    # 使用model自带的compile方法，选择预定义好的二分类交叉熵损失函数，Adam优化方法，以及准确率评估指标.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 导入作图工具包matplotlib
import matplotlib.pyplot as plt

# batch_size是每次进行参数更新的样本数量
batch_size = 32

# epochs将全部数据遍历训练的次数
epochs = 40 #40

from keras.callbacks import EarlyStopping
def model_fit(model:Sequential, x_train, y_train):
    # 定义早停回调函数
    early_stopping = EarlyStopping(monitor='val_loss',  # 监控验证集损失
                                   patience=5,  # 容忍连续5个epoch验证集损失不下降
                                   mode='min',  # 希望监控的指标越小越好
                                   restore_best_weights=True)  # 恢复到最佳权重

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    """用于模型训练"""
    # validation_split表示将全部训练数据的多少划分为验证集.
    #history =model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,callbacks=[early_stopping])
    history = model.fit(x_train, y_train,validation_data=(x_test,y_test), batch_size=batch_size, epochs=epochs,
                        callbacks=[early_stopping])
    return history


def plot_loss_acc(history, acc_png_path, loss_png_path):
    """用于绘制模型的损失和acc对照曲线, 以模型训练历史为参数"""
    # 首先获得模型训练历史字典，
    # 形如{'val_loss': [0.8132099324259264, ..., 0.8765081824927494],
    #    'val_acc': [0.029094827586206896,...,0.13038793103448276],
    #     'loss': [0.6650978644232184,..., 0.5267722122513928],
    #     'acc': [0.5803400383141762, ...,0.8469827586206896]}
    history_dict = history.history
    # 取出需要的各个key对应的value，准备作为纵坐标
    # acc = history_dict["acc"]
    acc = history_dict['accuracy']
    # val_acc = history_dict["val_acc"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    # 取epochs的递增列表作为横坐标
    epochs = range(1, len(acc) + 1)
    plt.switch_backend('Agg')
    # 绘制训练准确率的点图
    plt.plot(epochs, acc, 'bo', label="Training acc")
    # 绘制验证准确率的线图
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    # 增加标题
    plt.title("Training and Validation accuracy")
    # 增加横坐标名字
    plt.xlabel("Epochs")
    # 增加纵坐标名字
    plt.ylabel("Accuracy")
    # 将上面的图放在一块画板中
    plt.legend()
    # 保存图片
    plt.savefig(acc_png_path)

    # 清空面板
    plt.clf()
    # 绘制训练损失的点图
    plt.plot(epochs, loss, "bo", label="Training loss")
    # 绘制验证损失的线图
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    # 添加标题
    plt.title("Training and Validation loss")
    # 添加横坐标名字
    plt.xlabel("Epochs")
    # 添加纵坐标名字
    plt.ylabel("Loss")
    # 把两张图放在一起
    plt.legend()
    # 保存图片
    plt.savefig(loss_png_path)

#模型保存与加载过
from keras.models import load_model

def model_save(save_path, model:Sequential):
    """模型保存函数"""
    # 使用model.save对模型进行保存.
    model.save(save_path)

def model_load_(path):
    """模型加载与预测函数"""
    # 使用load_model方法进行加载
    return Sequential(load_model(os.path.join(user_path,path,'model.h5')))

def model_pred(model:Sequential,sample):
    # 使用predict方法进行预测
    result = model.predict(sample)
    return result

def get_x_test(path:str,cut_num,cutlen):
    x_train, y_train = word_map("./" + path + "/sample.csv", "./" + path + "/Tokenizer", cut_num)

    x_train = padding(x_train, cutlen)
    input_list = [1, 4, 9, 4, 1, 4]
    create_ngram_set(input_list, ngram_value=2)

    ti_path = "./" + path + "/token_indice"
    token_indice, new_max_features = get_ti_and_nmf(x_train, ti_path, ngram_range=2)

    x_train = add_ngram(x_train, token_indice, ngram_range=2)

    x_train, maxlen = align(x_train)  # 向量对齐
    return x_train

def build_pred(path:str,x_train,maxlen):
    t = joblib.load(os.path.join(user_path,path,"Tokenizer"))
    x_train = api.handle_cn_text(x_train)
    print('分词结果--->',x_train)
    x_train = t.texts_to_sequences([x_train]) #切词，词汇映射
    x_train = padding(x_train, cutlen) #文本补齐
    # 获得n-gram映射文件
    with open(os.path.join(user_path,path,"token_indice"), "r",encoding='utf-8') as f:
        token_indice = eval(f.read())
    x_train = add_ngram(x_train,token_indice , ngram_range) #ngram
    # 进行最大长度对齐
    x_train = padding(x_train, maxlen)
    #x_train = align(x_train)
    return x_train


def build_model(path:str,cut_num,cutlen):
    # word_map_()#文本拟合器
    x_train, y_train = word_map("./"+path+"/sample.csv", "./"+path+"/Tokenizer", cut_num)
    # print(padding(train[0],60))
    x_train = padding(x_train, cutlen)
    input_list = [1, 4, 9, 4, 1, 4]
    create_ngram_set(input_list, ngram_value=2)
    #print(create_ngram_set(input_list, ngram_value=2))
    # 2-gram特征组成的集合
    # {(4, 1), (9, 4), (4, 9), (1, 4)}

    # 数据进行截断对齐后的矩阵x_train
    # token_indice的保存路径
    ti_path = "./"+path+"/token_indice"
    token_indice, new_max_features = get_ti_and_nmf(x_train, ti_path, ngram_range=2)
    # token_indice 2-gram特征对应的数值
    '''
    {(28, 1329): 143282, (413, 841): 143283, 
    (8731, 6757): 143284, (4975, 68): 143285, 
    (581, 9339): 143286, (744, 1819): 143287, 
    (16, 1368): 143288, (17661, 4177): 143289,
     (20, 76): 143290, (495, 418): 143291, ...}
   '''
    # new_max_features 新的最大特征数
    # 143307
    x_train = add_ngram(x_train, token_indice, ngram_range=2)
    print(x_train)

    x_train, maxlen = align(x_train) #向量对齐
    print(x_train)
    print(maxlen)

    model = model_build(maxlen, new_max_features) #模型构造
    print(model)

    model = model_compile(model) #模型编译
    print(model)

    history = model_fit(model, x_train, y_train) #模型训练

    acc_png_path = "./"+path+"/acc.png"
    loss_png_path = "./"+path+"/loss.png"
    plot_loss_acc(history, acc_png_path, loss_png_path) #目标评估

    # 模型的保存路径
    save_path = "./"+path+"/model.h5"
    # 训练之后的model对象
    model_save(save_path, model)



if __name__=='__main__':
    #model_build_()
    build_model('movie', 43000*2, cutlen)
    build_model('fashion', 15000*2, cutlen)
    #build_model('star', 7800*2, cutlen)
    #build_model('beauty',26037*2,cutlen)

    #save_path = "./movie/model.h5"
    #sample = np.array([get_x_test('movie', 43000*2, cutlen)[0]])
    #result = model_pred(model_load_(save_path),sample)
   # print(result)


