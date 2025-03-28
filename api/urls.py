from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('api/first_show', views.first_show),
    path('api/get_cache', views.get_cache),
    path('api/get_recomm', views.get_recomm),
    path('ctone/api/test', views.test),
    path('api/get_label',views.get_label), #获取标签

    #text ="我的眼睛很大很大,可以装得下天空，装得下高山，装得下大海，装得下整个世界；我的眼睛又很小很小，有心事时，就连两行眼泪，也装不下."
    path('api/cntext',views.handle_cn_text), #切词
    #words= ['眼睛', '很大', '很大', '装得', '天空', '装得', '高山', '装得', '大海', '装得', '整个', '世界', '眼睛', '很小', '很小', '心事', '两行', '眼泪', '装不下']
    path('api/index/label',views.get_index_map_label),#获取标签权重
    #words = ['眼睛', '很大', '很大', '装得', '天空', '装得', '高山', '装得', '大海', '装得', '整个', '世界', '眼睛', '很小', '很小', '心事', '两行', '眼泪', '装不下']
    #indexLabel=["0", [["美妆", 0.654], ["情感故事", 0.765]]]
    path('api/weight/update/',views.weight_update),#更新标签权重，从模型获取预测的概率
    #index_map_label = ["2", [["情感故事", 0.765]], "3", [["情感故事",  0.876], ["明星", 0.765]]]
    path('api/control/increase/', views.control_increase),#标签概率调整处理
    #index_map_label = ["2", [["情感故事", 0.765]], "3", [["情感故事",  0.876], ["明星", 0.765]]]
    path('api/father/normalized/', views.father_label_and_normalized),#标签概率归一化处理
    path('api/create/label/rel/', views.create_label_node_and_rel),#创建标签和关系
    path('api/find/label/rel/', views.get_all_node_rel),  # 查询标签和关系
    path('api/label/list/', views.get_label_list),  # 获取标签列表
    path('api/label/create/words/', views.create_labels_words),  # 创建标签所有的词汇
    path('api/vocabulary/node/rel/', views.create_vocabulary_node_and_rel),  # 创建标签关系
    path('api/text/list/', views.get_p_text_list),  #测试获取单个文章列表
    path('api/p/sample/', views.get_p_sample),  #生成单个所有正样本
    path('api/all/sample/', views.get_p_sample_),  # 生成所有正样本
    path('api/single/sample/', views.get_sample_),  #生成所有正负样本
    path('api/train/data/labels/', views.get_data_labels_), #生成所有训练数据
    path('api/labels/distribution/', views.get_labels_distribution_), #所有标签正负样本数量的基本分布情况
    path('api/length/distribution/', views.get_sentence_length_distribution_), #所有标签正负样本句子长度基本分布情况
    path('api/frequency/distribution/', views.get_word_frequency_distribution_),  # 所有标签正负样本词频基本分布情况





]
