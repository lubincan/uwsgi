import sys


import time
from keras import backend as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import tensorflow as tf
# 关闭急切执行
#tf.compat.v1.disable_eager_execution()

import os
user_path=os.path.dirname(__file__)

def request_model_serve(word_list, label_list):
    return [["情感故事", 0.865]]
def to_savedmodel(h5_model_path, pb_model_path):
    """将h5模型转化成tensorflow的pb格式模型"""
    # 处理文件路径编码
    h5_model_path = os.path.normpath(os.path.join(user_path, h5_model_path))
    pb_model_path = os.path.normpath(os.path.join(user_path, pb_model_path))
    # 创建目录（如果不存在）
    if not os.path.exists(pb_model_path):
        os.makedirs(pb_model_path)
    # 加载Keras模型
    model = load_model(h5_model_path)

    # 保存模型为 SavedModel 格式
    tf.saved_model.save(model, pb_model_path)

    # 创建SavedModel构建器
    '''builder = saved_model_builder.SavedModelBuilder(pb_model_path)

    # 定义模型的预测签名
    '''
      #使用predict_signature_def函数定义模型的预测签名。签名是SavedModel的重要组成部分，它描述了模型的输入输出接口。
      #inputs：是一个字典，键'input'是输入的名称，值model.inputs[0]是模型的第一个输入张量。
      #outputs：是一个字典，键'income'是输出的名称，值model.outputs[0]是模型的第一个输出张量
    '''
    signature = predict_signature_def(
        inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

    # 获取会话并保存模型
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()
'''
def to_savedmodel_(path:str):
    h5_model_path = "./"+path+"/model.h5"
    time_ = str(int(time.time()))
    pb_model_path = "./"+path+"/" + time_

    to_savedmodel(h5_model_path, pb_model_path)

# 导入必备的工具包
import json
import threading
import requests
import joblib

# 从任意的模型训练文件中导入add_ngram增加n-gram特征以及padding截断函数
import  text_labeled.model_train.model_all_train as train
# 定义模型配置路径，它指向一个json文件
model_config_path = os.path.join(user_path,"model_config.json")

# model_config.json形如 ：
# {"影视": ["/data/django-uwsgi/text_labeled/model_train/movie/Tokenizer", 60, 2,
#           "/data/django-uwsgi/text_labeled/model_train/movie/token_indice", 119,
#           "http://localhost:8501/v1/models/movie:predict"],
# "美妆": ["/data/django-uwsgi/text_labeled/model_train/beauty/Tokenizer", 75, 2,
#           "/data/django-uwsgi/text_labeled/model_train/beauty/token_indice", 119,
#           "http://localhost:8502/v1/models/beauty:predict"]}
# json文件中是一个字典，字典中的每个key是我们标签的中文字符，每个value是一个列表，
# 列表的第一项是特征处理词汇映射器的存储地址
# 第二项是特征处理语料的截断长度
# 第三项是n-gram取得n值
# 第四项是n-gram特征中token_indice的保存路径
# 第五项是最后的最大的对齐长度
# 第六项是该模型对应的微服务地址

# 最终的模型预测结果列表
model_prediction = []


def fea_process(word_list, config_list):
    """对输入进行类似与训练前的特征处理过程"""
    # 读取设定好的配置
    tokenizer_path = config_list[0]
    cutlen = config_list[1]
    ngram_range = config_list[2]
    ti_path = config_list[3]
    maxlen = config_list[4]

    # 加载分词映射器
    t = joblib.load(tokenizer_path)
    x_train = t.texts_to_sequences([word_list])
    # 进行截断对齐
    #x_train = train.padding(x_train, cutlen)
    x_train = train.align(x_train)
    # 获得n-gram映射文件
    with open(ti_path, "r",encoding='utf-8') as f:
        token_indice = eval(f.read())
    # 添加n-gram特征
    x_train = train.add_ngram(x_train, token_indice, ngram_range)
    # 进行最大长度对齐
    #x_train = train.padding(x_train, maxlen)
    x_train = train.align(x_train)
    return x_train

def pred(path, x_train,maxlen,url):
    x_train = train.build_pred(path, x_train,maxlen)
    # 封装成tf-serving需要的数据体
    data = {"instances": x_train.tolist()}
    # 向刚刚封装的微服务发送请求
    res = requests.post(url=url, json=data)
    # 将该线程中获取的结果放到模型预测结果列表中
    model_prediction.append([path, eval(res.text)["predictions"][0][0]])
    return res.text

def request_model_serve_thread(p_u:dict,x_train, maxlen):
    model_prediction.clear() #清空
    """该函数开启多线程请求封装好的模型微服务"""
    def _start_thread(pred, path, x_train,maxlen,url):
        """开启预测线程, 以线程需要执行的函数和函数的输入为参数"""
        t = threading.Thread(target=pred, args=(path, x_train,maxlen,url))
        t.start()
        return t

    # 遍历model_list, 调用开启线程函数_start_thread，会获得一个所有开启后的线程列表
    t_list = list(map(lambda path: _start_thread(pred, path, x_train,maxlen,p_u[path]), p_u.keys()))
    # 线程将逐一join操作等待所有线程完成
    t_list = list(map(lambda t: t.join(), t_list))
    # 最后过滤掉所有概率预测小于0.5的类别，返回结果
    result = list(filter(lambda x: x[1] >= 0.5, model_prediction))
    return result

def predict_test_h5(x_tr):
    result = []
    word_list = ["霸王别姬", "是一部", "非常", "值得", "看的", "电影"]
    model = "影视"
    model = train.model_load_("movie")
    print('movie--->',model.input_shape)
    #x_tr = "影联传媒曾先后发行《西游记之大圣归来》、《战狼Ⅱ》、《我不是药神》、《流浪地球》等优质影片"
    x_train = train.build_pred("movie", x_tr, model.input_shape[1])
    print(x_train)
    res = train.model_pred(model, x_train)
    result.append("电影")
    result.append(str(res[0][0]))
    print(res)
    # print('------->',pred(word_list, model))

    model = train.model_load_("star")
    print('star--->',model.input_shape)
    #x_tr = "划重点：.热巴签约了十五年，这一个女演员的最好的时期了耗在这儿了"
    x_train = train.build_pred("star", x_tr, model.input_shape[1])
    print(x_train)
    res = train.model_pred(model, x_train)
    result.append("明星")
    result.append(str(res[0][0]))
    print(res)


    model = train.model_load_("beauty")
    print('beauty--->',model.input_shape)
    x_train = train.build_pred("beauty", x_tr, model.input_shape[1])
    print(x_train)
    res = train.model_pred(model, x_train)
    result.append("美妆")
    result.append(str(res[0][0]))
    print(res)

    model = train.model_load_("fashion")
    print('fashion--->',model.input_shape[1])
    x_train = train.build_pred("fashion", x_tr, model.input_shape[1])
    print(x_train)
    res = train.model_pred(model, x_train)
    print(res)
    result.append("时尚")
    result.append(str(res[0][0]))
    return result

def predict_test_pb():
    '''
    模型所在目录结构，不过目录不对请自己调整
        /models
        └── fashion
            └── 1
                ├── saved_model.pb
                └── variables
                    ├── variables.data-00000-of-00001
                    └── variables.index

    docker启动脚本

    docker run -t --rm --name movie -p 8501:8501 -v "D:/pythonwork/uwsgi/text_labeled/model_train/movie:/models/movie" -e MODEL_NAME=movie tensorflow/serving &

    docker run -t --rm --name beauty -p 8502:8501 -v "D:/pythonwork/uwsgi/text_labeled/model_train/beauty:/models/beauty" -e MODEL_NAME=beauty tensorflow/serving &

    docker run -t --rm --name star -p 8503:8501 -v "D:/pythonwork/uwsgi/text_labeled/model_train/star:/models/star" -e MODEL_NAME=star tensorflow/serving &

    docker run -t --rm --name fashion -p 8504:8501 -v "D:/pythonwork/uwsgi/text_labeled/model_train/fashion/:/models/fashion" -e MODEL_NAME=fashion tensorflow/serving &
    '''

    x_tr = "影联传媒曾先后发行《西游记之大圣归来》、《战狼Ⅱ》、《我不是药神》、《流浪地球》等优质影片"
    result = pred('movie', x_tr, 119, 'http://localhost:8501/v1/models/movie:predict')
    print('movie--->', result)

    result = pred('star', x_tr, 119, 'http://localhost:8503/v1/models/star:predict')
    print('star---->', result)

    result = pred('fashion', x_tr, 109, 'http://localhost:8504/v1/models/fashion:predict')
    print('fashion--->', result)

    result = pred('beauty', x_tr, 75, 'http://localhost:8502/v1/models/beauty:predict')
    print('beauty----->', result)

if __name__ == "__main__":
    x_tr = "影联传媒曾先后发行《西游记之大圣归来》、《战狼Ⅱ》、《我不是药神》、《流浪地球》等优质影片"
    #to_savedmodel_('beauty') #将h5格式转换为SavedModel格式模型
    #to_savedmodel_('movie') #将h5格式转换为SavedModel格式模型
    #to_savedmodel_('fashion') #将h5格式转换为SavedModel格式模型
    #to_savedmodel_('star') #将h5格式转换为SavedModel格式模型
    # 分词列表
    result_h5 = predict_test_h5(x_tr)#基于h5结构的预测
    print('居于h5的预测',result_h5)
    #predict_test_pb()
    #print(model_prediction)
    #x_tr = "影联传媒曾先后发行《西游记之大圣归来》、《战狼Ⅱ》、《我不是药神》、《流浪地球》等优质影片"
    #p_u={"movie":"http://localhost:8501/v1/models/movie:predict","star":"http://localhost:8503/v1/models/star:predict","fashion":"http://localhost:8504/v1/models/fashion:predict","beauty":"http://localhost:8502/v1/models/beauty:predict"}
    #result = request_model_serve_thread(p_u,x_tr,119)
    #print(result)
