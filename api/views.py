import json

from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from django.http import HttpResponse
from text_labeled import api
from text_labeled.create_graph import build,get_vocab
from text_labeled.settings import LABEL_STRUCTURE
from text_labeled.model_train import get_sample as smp,data_analysis as analysis
import text_labeled.model_train.multithread_predict as predict
#@api_view(["GET"])
def test(request):
    print("---------9999")
    return HttpResponse("This is a test.")

# 该装饰器用于保证函数能够接收POST请求
@api_view(['POST','GET'])
def get_label(request):
    """获取标签接口, 参数request是请求体, 包含前端传来的数据"""
    # 通过请求体接收前端传来的数据text
    #request.POST.get("text")
    text = request.POST.get('text')
    # 调用text_labeled/api.py文件中的label函数进行处理
    #result = api.label(text)
    result = predict.predict_test_h5(text)
    #result = get_label_neo4j(text)
    # 返回json格式的结果，并使用HttpResponse进行封装
    return HttpResponse(json.dumps(result,ensure_ascii=False),charset='utf-8')

def get_label_neo4j(text):
    # 接收POST请求，并取数据中的"text"对应的值
    #text = request.POST.get("text")
    # 开始调用text_labeled目录下的api.py中的函数
    # 调用输入预处理
    word_list = api.handle_cn_text(text)
    # 调用图谱匹配
    index_map_label = api.get_index_map_label(word_list)
    # 调用匹配歧义判断
    index_map_label_ = api.weight_update(word_list, index_map_label)
    if index_map_label_:
        # 调用概率调整
        df_ = api.control_increase(index_map_label_)
        # 调用概率归一化与父标签检索
        result = api.father_label_and_normalized(df_)
    else:
        result = []
    # 然后封装成响应体返回结果
    return result
@api_view(['GET'])
def handle_cn_text(request):
    return HttpResponse(json.dumps(api.handle_cn_text(request.GET.get('text')),ensure_ascii=False),charset='utf-8');
@api_view(['POST'])
def get_index_map_label(request):
    word_list = request.POST.get("words")
    return HttpResponse(json.dumps(api.get_index_map_label(word_list),ensure_ascii=FloatingPointError),charset='utf-8');

@api_view(['POST'])
def weight_update(request):
    word_list = request.POST.get("words")
    index_map_label = request.POST.get("indexLabel")
    return HttpResponse(json.dumps(api.weight_update(word_list,index_map_label),ensure_ascii=False),charset='utf-8')

@api_view(['POST'])
def control_increase(request):
    return HttpResponse(json.dumps(api.control_increase(request.POST.get('index_map_label')), ensure_ascii=False), charset='utf-8')

@api_view(['POST'])
def father_label_and_normalized(request):
    return HttpResponse(json.dumps(api.father_label_and_normalized(api.control_increase(request.POST.get('index_map_label'))), ensure_ascii=False),
                        charset='utf-8')
@api_view(['GET'])
def create_label_node_and_rel(request):
    build.create_label_node_and_rel()
    return HttpResponse(json.dumps(build.get_all_node_rel(),ensure_ascii=False),charset='utf-8')
@api_view(['GET'])
def get_all_node_rel(request):
    return HttpResponse(json.dumps(build.get_all_node_rel(),ensure_ascii=False),charset='utf-8')

@api_view(['GET'])
def get_label_list(request):
    return HttpResponse(json.dumps(LABEL_STRUCTURE,ensure_ascii=False),charset='utf-8')

@api_view(['GET'])
def create_labels_words(request):
    get_vocab.create_labels_words()
    return HttpResponse(json.dumps("已全部写入标签词汇",ensure_ascii=False),charset='utf-8')

@api_view(['GET'])
def create_vocabulary_node_and_rel():
    build.create_vocabulary_node_and_rel()
    return HttpResponse(json.dumps("已全部创建标签关系",ensure_ascii=False),charset='utf-8')

@api_view(['GET'])
def get_p_text_list(request):
    print('-------------------')
    return HttpResponse(json.dumps(smp.get_p_text_list_test(),ensure_ascii=FloatingPointError),charset='utf-8')

@api_view(['GET'])
def get_p_sample(request):
    smp.get_p_sample("../create_graph/beauty/","./beauty")
    return HttpResponse(json.dumps("已经全部执行完毕", ensure_ascii=False), charset='utf-8')

@api_view(['GET'])
def get_p_sample_(request):
    smp.get_p_sample_()
    return HttpResponse(json.dumps("所有分类标签正样本已全部生成", ensure_ascii=False), charset='utf-8')
@api_view(['GET'])
def get_sample_(request):
    smp.get_sample_()
    return HttpResponse(json.dumps("所有分类标签正样本和负样本已全部生成", ensure_ascii=False), charset='utf-8')

@api_view(['GET'])
def get_data_labels_(request):
    return HttpResponse(analysis.get_data_labels_(), charset='utf-8')

@api_view(['GET'])
def get_labels_distribution_(request):

    return HttpResponse(analysis.get_labels_distribution_(), charset='utf-8')
@api_view(['GET'])
def get_sentence_length_distribution_(request):
    return HttpResponse(analysis.get_sentence_length_distribution_(), charset='utf-8')

@api_view(['GET'])
def get_word_frequency_distribution_(request):
    return HttpResponse(analysis.get_word_frequency_distribution_(), charset='utf-8')
