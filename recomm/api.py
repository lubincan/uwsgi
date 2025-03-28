from neo4j import GraphDatabase
#from neo4j.v1 import GraphDatabase
import random
from itertools import chain
import redis
import cv2
import requests
import time
import os
user_path=os.path.dirname(__file__)
NEO4J_CONFIG = dict({
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "password"),
    "encrypted": False
})

_driver = GraphDatabase.driver(**NEO4J_CONFIG)

REDIS_CONFIG = dict({
    "host":"127.0.0.1",
    "port":"6379",
    "decode_responses":True
})

pool = redis.ConnectionPool(**REDIS_CONFIG)


def _get_hot():
    """获得热门召回推荐"""
    # 2*评论数量+喜欢数量+3*转发
    with _driver.session() as session:
        cypher = "match(a:SuperfansPost) set a.hot_score = 2 * a.commented_num + a.liked_num + 3*a.forwarded_num return a.pid order by a.hot_score desc limit 100"
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))
    return result


def get_last():
    """获得时间召回推荐"""
    with _driver.session() as session:
        cypher = "match(a:SuperfansPost) return a.pid order by a.publish_time desc limit 100"
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))
    return result


def get_v():
    """获得速度召回推荐"""
    with _driver.session() as session:
        cypher = "match(a:SuperfansPost_A) match(b:SuperfansPost) where(a.pid = b.pid) set b.v=a.hot_sore - b.hot_score return b.pid order by b.v desc"
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))
    return result


def get_r(uid):
    """获得基于用户的帖子二度关系召回"""
    # uid-->b帖子-->c用户--->d帖子
    with _driver.session() as session:
        cypher = "match(a{uid:%d})-[r]-(b:SuperfansPost)-[r2]-(c:SuperfansUser)-[r3]-(d:SuperfansPost) return d.pid limit 100" % int(uid)
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))
    return result


def get_random():
    """获得运营自定义的随机召回结果"""
    with _driver.session() as session:
        cypher = "match(a:SuperfansPost) return a.pid limit 1000"
        record = session.run(cypher)
        pid_list = list(map(lambda x: x[0], record))
    result = random.sample(pid_list, 20)
    return result


def pyramid_array(all_data):
    result = []
    for pid in set(chain(*all_data)):
        v = 0
        for list_ in all_data:
            if pid in list_:
                v += 1
        result.append([pid, v])

    result.sort(key=lambda x: x[1])
    return list(map(lambda x: x[0], result))[::-1]


def j_data_write(uid, j_data):
    """将该用户的所有金字塔数据（金字塔结构处理的）pid 部分写入到缓存
    :param uid: 请求用户id
    :param j_data: pyramid_array处理之后的结果
    :return: r_data
    """
    # 建立redis连接
    r = redis.StrictRedis(connection_pool=pool)
    r.set('j_' + str(uid), str(j_data))

    # 判断有误历史缓存结果
    old = r.get('r_' + str(uid))
    if not old:
        r_data = j_data[:50]
    else:
        r_data = list(set(j_data) - set(eval(old)))[:50]

    r.set('r_' + str(uid), str(r_data))
    return r_data


def get_img_fingerprints(gray_dct_ul64_list, gray_dct_ul64_avg):
    """
    获取图片指纹：遍历灰度图左上8*8的所有像素，比平均值大则记录为1，否则记录为0。
    :param gray_dct_ul64_list: 灰度图左上8*8的所有像素
    :param gray_dct_ul64_avg: 灰度图左上8*8的所有像素平均值
    :return: 图片指纹
    """
    img_fingerprints = ''
    avg = gray_dct_ul64_avg[0]
    for i in range(8):
        for j in range(8):
            if gray_dct_ul64_list[i][j] > avg:
                img_fingerprints += '1'
            else:
                img_fingerprints += '0'
    return img_fingerprints


def get_img_gray_bit(img, resize=(32, 32)):
    """
    获取图片指纹
    :param img: 图片
    :param resize: Resize的图片大小
    :return: 图片指纹
    """
    # 修改图片大小
    image_resize = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
    # 修改图片成灰度图
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    # 转换灰度图成浮点型
    image_gray_f = np.float32(image_gray)
    # 获取灰度图的DCT集合
    image_gray_dct = cv2.dct(image_gray_f)
    # 获取灰度图DCT集合的左上角8*8
    # gray_dct_ul64_list = get_gray_dct_ul64_list(image_gray_dct)
    gray_dct_ul64_list = image_gray_dct[0:8, 0:8]
    # 获取灰度图DCT集合的左上角8*8对应的平均值
    # gray_dct_ul64_avg = get_gray_dct_ul64_avg(gray_dct_ul64_list)
    gray_dct_ul64_avg = cv2.mean(gray_dct_ul64_list)
    # 获取图片指纹
    img_fingerprints = get_img_fingerprints(
        gray_dct_ul64_list, gray_dct_ul64_avg)
    return img_fingerprints


def get_mh(img_fingerprints1, img_fingerprints2):
    """
    获取汉明距离
    :param img_fingerprints1: 比较对象1的指纹
    :param img_fingerprints2: 比较对象2的指纹
    :return: 汉明距离
    """
    hm = 0
    for i in range(0, len(img_fingerprints1)):
        if img_fingerprints1[i] != img_fingerprints2[i]:
            hm += 1
    return hm


def is_image_file(file_name):
    """
    判断文件是否是图片
    :param file_name: 文件名称(包含后缀信息)
    :return: 1:图片，0:非图片
    """
    ext = (os.path.splitext(file_name)[1]).lower()
    if ext == ".jpg" or ext == ".jpeg" or ext == ".bmp" or ext == ".png":
        return 1
    return 0


def get_all_img_list(root_path):
    """
    获取目标文件夹下所有图片路径集合
    :param root_path: 目标文件夹
    :return: 图片集合
    """
    img_list = []
    # 获取目标文件夹下所有元组
    root = os.walk(root_path)
    # 循环元组，获取目标文件夹下所有图片路径集合
    for objects in root:
        for obj in objects:
            if "/" in str(obj):
                # 记录文件夹路径
                path = str(obj)
            elif len(obj) > 0:
                # 如果是文件，判断是否是图片。如果是图片则保存进
                for file in obj:
                    if "." in str(file) and is_image_file(file) == 1:
                        full_path = path + "/" + str(file)
                        img_list.append(full_path.replace("\\", "/"))
    return img_list


def compare_img(root_path):
    """
    比较图片 (Main)
    :param root_path: 目标文件夹
    """
    compared_list = []
    # 获取目标文件夹下所有图片路径集合
    img_list = get_all_img_list(root_path)
    # 遍历目标文件夹下所有图片进行两两比较
    for file1 in img_list:
        # 已经发现是相同的图片不再比较
        if file1 in compared_list:
            continue
        # im1 = cv2.imread(files1)
        im1 = cv2.imdecode(
            np.fromfile(
                file1,
                dtype=np.uint8),
            cv2.IMREAD_UNCHANGED)
        print(im1)
        if im1 is None:
            continue
        im1_size = os.path.getsize(file1)
        img_fingerprints1 = get_img_gray_bit(im1)
        print("第一张的指纹:", img_fingerprints1)
        for file2 in img_list:
            if file1 != file2:
                # im2 = cv2.imread(files2)
                im2 = cv2.imdecode(
                    np.fromfile(
                        file2,
                        dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED)
                if im2 is None:
                    continue
                im2_size = os.path.getsize(file2)
                print(im2_size)
                # 如果两张图片字节数大小一样再判断汉明距离
                if im1_size != im2_size:
                    img_fingerprints2 = get_img_gray_bit(im2)
                    print("第二张的指纹:", img_fingerprints2)
                    compare_result = get_mh(
                        img_fingerprints1, img_fingerprints2)
                    print(compare_result)
                    # 汉明距离等于0，说明两张图片完全一样
                    if compare_result == 0:

                        compared_list.append(file2)
                        compared_list.append(file1)
    return compared_list


def d_hash_serve(url_list):
    # 下载所有图片
    default_image = "test.png"
    path = os.path.join(user_path, "img_compare/")
    def download(url):

        try:
            img = requests.get(url)
            #if not img:
                #print('------------->', path + url)
            with open(path + url + ".jpg", "wb") as fp:

                fp.write(img.text)
            return path + url + ".jpg"
        except BaseException:
            return path + default_image

    # 得到下载后的图片路径列表
    url_path = list(map(lambda url: download(url), url_list))

    # 得到本地路径与CDN路径的对应字典
    url_dict = dict(zip(url_path, url_list))

    # 对该目录下的图片文件进行比较
    compare_list = compare_img(path)

    # 获得比较之后应该删除的重复图片CDN路径
    compared_delete_list = list(
        map(lambda x: url_dict[x], list(set(compare_list[::2]))))

    # 根据CDN路径进行列表元素删除
    list(map(lambda x: url_list.remove(x), compared_delete_list))

    return url_list


def rfilter(r_data):
    """规则过滤器(服务)函数"""

    with _driver.session() as session:
        # 同批次数据进行对比-找到图片相似的帖子
        def get_url(pid):
            cypher = "match(a:SuperfansPost{pid:%d}) return a.iv_url" % (pid)
            record = session.run(cypher)
            result = list(map(lambda x: x[0], record))
            if result:
                return result[0]
        url_list = list(
            filter(
                lambda x: x is not None, map(
                    lambda x: get_url(x), r_data)))

    url_dict = dict(zip(url_list, r_data))
    url_list = d_hash_serve(url_list)
    result = list(map(lambda x: url_dict[x], url_list))
    return result


def _get_recomm(IP, uid):
    """整体推荐流程
    :param IP: 用户的IP
    :param uid: 用户ID
    :return:
    """
    # 1、召回的模块接口实现
    #（1）获得热门召回数据
    hot_data = _get_hot()
    # 获得最近发布召回数据
    last_data = get_last()
    # 获得单位时间内增长速度最快的帖子
    v_data = get_v()
    # 获得基于用户的协同过滤召回数据
    r_data = get_r(uid)
    # 随机召回数据
    random_data = get_random()
    # 合并召回结果
    all_data = [hot_data] + [last_data] + [v_data] + [r_data] + [random_data]
    # 金字塔结构实现，给PID分配权重排序输出pid
    j_data = pyramid_array(all_data)
    # 将数据写入金字塔缓存
    r_data = j_data_write(uid, j_data)
    # 规则过滤器
    f_data = rfilter(r_data)

    return f_data


def v_get_cache(IP):
    """用户获取请求视图的缓存读取逻辑
    :param IP: 用户请求IP
    :return:将推荐的结果交给大模型预测，看正样本比例多分，然后取得分高的帖子推荐给用户
    """
    # redis连接
    r = redis.StrictRedis(connection_pool=pool)
    # 获取这个用户IP，判断这个IP是否存在
    uid = r.get("u_" + IP)
    if not uid:
        # 随机设置一个UID
        uid = random.choice([10033736])
        r.set("u_" + IP, uid)
    # 读取该用户UID的缓存结果
    # 如果没有
    # _get_recomm
    if not r.llen(IP):
        print('----------++++')
        return _get_recomm(IP, int(uid))
    else:
        # 有
        # 读取缓存推荐出去
        pid_list = eval(r.lpop(IP))

        # 取出PID对应的相关属性
        with _driver.session() as session:
            cypher = "match(a:SuperfansPost{pid:%d}) return properties(a)"
            # 处理读取结果
            record = list(map(lambda x: session.run(cypher % x), pid_list))
            # 取出属性和id
            result = list(map(lambda y: list(map(lambda x: x[0], y))[0], record))
        return result


# 点赞
cypher1 = "MATCH(a:{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b MERGE(a)-[r:%s]-(b)"
# 评论
cypher2 = "MATCH(a:SuperfansUser{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b CREATE(a)-[r:%s]-(b) set r.time=%d, r.content=%s"
# 转发
cypher3 = "MATCH(a:SuperfansUser{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b CREATE(a)-[r:%s]-(b)"
# 取消点赞
cypher4 = "MATCH(a:SuperfansUser{uid:%d})-[r:%s]-(b:SuperfansPost{pid:%d}) delete r"
# 删除评论
cypher5 = "MATCH(a:SuperfansUser{uid:%d})-[r:%s]-(b:SuperfansPost{pid:%d}) where r.time=%d delete r"

r_result = {"msg":"Success", "code": 1}
f_result = {"msg": "Fail", "code": 0}


def write_to_neo4j(IP, pid, type_, content=""):
    """写入行为到neo4j数据库
    :param IP: 用户IP
    :param pid: 帖子ID
    :param type_: 行为类型
    :param content: 评论内容
    :return:
    """
    # 1、读取redis缓存，是否含有这个用户
    r = redis.StrictRedis(connection_pool=pool)
    uid = r.get("u_" + IP)

    # 行为类型判断，执行不同cypher语句
    if type_ == "like":
        if uid:
            with _driver.session() as session:
                session.run(cypher1 % (int(uid), int(pid)), type_)
                return r_result
        else:
            return f_result
    elif type_ == "comment":
        time_ = int(time.time())
        if uid:
            with _driver.session(cypher2 % (int(uid), int(pid), type_, time_, content)) as session:
                session.run()
                return r_result
        else:
            return f_result
    else:
        if uid:
            with _driver.session() as session:
                session.run(cypher3 % (int(uid), int(pid)), type_)
                return r_result
        else:
            return f_result


def cancel_to_neo4j(IP, pid, type_, time_):
    """
    删除行为类型
    :param IP: 用户IP
    :param pid: 帖子ID
    :param type_: 行为类型
    :param time_: 时间戳
    :return:
    """
    r = redis.StrictRedis(connection_pool=pool)
    uid = r.get("u_" + IP)
    if type_ == "like":
        if uid:
            with _driver.session() as session:
                session.run(cypher4 %(int(uid), type_, int(pid)))
                return r_result
        else:
            return f_result
    # 如果是评论取消
    elif type_ == "comment":
        if uid:
            # 根据时间戳删除
            with _driver.session() as session:
                session.run(cypher5 %(int(uid),type_, int(pid), time_))
                return r_result
        else:
            return f_result





