import os
import sys
#sys.path.append('../../')
# 限制句子的最小字符数和句子的最大字符数
MIN_LENGTH = 5
MAX_LENGTH = 500
user_path=os.path.dirname(__file__)

def get_p_text_list(single_article_path):
    """获取单篇文章的文本列表"""
    print(single_article_path)
    if not os.path.exists(single_article_path):
        print(f"文件不存在: {single_article_path}")
        return []
    with open(os.path.abspath(single_article_path), "r",encoding='utf-8') as f:
        text = f.read()
        # 去掉换行符, 并以句号划分
        cl = text.replace('\n','.').split("。")
        # 过滤掉长度范围之外的句子
        cl = list(filter(lambda x: MIN_LENGTH<len(x) < MAX_LENGTH,cl))
    return cl

def get_p_text_list_test():
    return get_p_text_list(os.getcwd().strip()+"\\text_labeled\\create_graph\\beauty\\article-183196")

# 原始语料在上一个目录的create_graph目录下
single_article_path = "../create_graph/beauty/article-191721"


def get_p_sample(a_path, p_path):

    """该函数用于获得正样本的csv, 以文章路径和正样本csv写入路径为参数"""
    #if not os.path.exists(os.path.join(user_path,a_path)): return
   # if not os.path.exists(p_path): os.mkdir(p_path)
    # 以追加的方式打开预写入正样本的csv
    fp = open(os.path.join(user_path,p_path,"p_sample.csv"), 'a',encoding='utf-8')
    # 遍历文章目录下的每一篇文章
    for u in os.listdir(os.path.join(user_path,a_path)):
        cl = get_p_text_list(os.path.join(user_path,a_path,u))
        for clc in cl:
            fp.write('1' + '\t' +clc + '\n')
    fp.close()




def get_sample(p_path, n_path_csv_list: list):
    """该函数用于获取样本集包括正负样本, 以正样本csv文件路径和负样本csv文件路径列表为参数"""
    fp = open(os.path.join(user_path,p_path,'sample.csv'), 'w', encoding='utf-8')
    with open(os.path.join(user_path,p_path,"p_sample.csv"),'r',encoding='utf-8') as f:
        text = f.read()
        # 先将正样本写入样本csv之中
        fp.write(text)
        # 遍历负样本的csv列表
        for n_p_c in n_path_csv_list:
            with open(os.path.join(user_path,n_p_c), 'r',encoding='utf-8') as f:
                #将其中的标签1改为0
                text = f.read().replace('1','0')
            #然后写入样本的csv中
            fp.write(text)
        fp.close()


# 原始语料在上一个目录的create_graph目录下

n_path_csv_list = ["./movie/p_sample.csv", "./star/p_sample.csv", "./fashion/p_sample.csv""./beauty/p_sample.csv"]
paths = ["../create_graph/beauty/", "./beauty", "../create_graph/fashion/", "./fashion", "../create_graph/movie/",
         "./movie", "../create_graph/star/", "./star"]


def get_p_sample_():
    print('------>',os.path.dirname(__file__))
    list(map(lambda x, y: get_p_sample(x, y), paths[0::2], paths[1::2]))

def get_sample_():

    list(map(lambda y: get_sample(y, n_path_csv_list), paths[1::2]))


if __name__ =='__main__':
    #pass

    #print(get_p_text_list(single_article_path))

    # 原始语料在上一个目录的create_graph目录下
    get_p_sample_()
    get_sample_()



    '''
    a_path = "../create_graph/beauty/"
    p_path = "./beauty"
    get_p_sample(a_path,p_path)
    get_sample(p_path, n_path_csv_list)
    # 原始语料在上一个目录的create_graph目录下
    a_path = "../create_graph/fashion/"
    p_path = "./fashion"
    get_p_sample(a_path, p_path)
    get_sample(p_path, n_path_csv_list)
    a_path = "../create_graph/movie/"
    p_path = "./movie"
    get_p_sample(a_path, p_path)
    get_sample(p_path, n_path_csv_list)
    a_path = "../create_graph/star/"
    p_path = "./star"
    get_p_sample(a_path, p_path)
    get_sample(p_path,n_path_csv_list)
    '''



