import csv
from neo4j import GraphDatabase
NEO4J_CONFIG = dict({
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "password"),
    #'encrpted': False,
    "encrypted": False
})


driver = GraphDatabase.driver(**NEO4J_CONFIG)

def import_user_nodes(file_path):
    #file_path = 'dm_user_profile_10.csv'
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with driver.session() as session:
            for line in reader:
                query = """
                CREATE (:SuperfansUser {
                    # 用户唯一标识，将字符串转换为整数类型
                    uid: $uid,
                    # 用户昵称，为字符串类型
                    nickname: $nickname,
                    # 用户使用的设备型号，为字符串类型
                    device_model: $device_model,
                    # 用户设备的操作系统，为字符串类型
                    device_system: $device_system,
                    # 用户关注的明星列表，将 CSV 中的字符串按逗号分割为列表
                    follow_stars_list: $follow_stars_list,
                    # 用户发布的帖子数量，将字符串转换为整数类型
                    publish_posts_num: $publish_posts_num,
                    # 用户点赞的帖子数量，将字符串转换为整数类型
                    like_posts_num: $like_posts_num,
                    # 用户评论的帖子数量，将字符串转换为整数类型
                    comment_posts_num: $comment_posts_num,
                    # 用户转发的帖子数量，将字符串转换为整数类型
                    forward_posts_num: $forward_posts_num,
                    # 用户举报的帖子数量，将字符串转换为整数类型
                    report_posts_num: $report_posts_num,
                    # 用户最后一次登录的时间戳，将字符串转换为整数类型
                    last_signin_time: $last_signin_time
                })
                """
                parameters = {
                    "uid": int(float(line[0])),
                    "nickname": line[1],
                    "device_model": line[2],
                    "device_system": line[3],
                    "follow_stars_list": line[4].split(","),
                    "publish_posts_num": int(float(line[5])),
                    "like_posts_num": int(float(line[6])),
                    "comment_posts_num": int(float(line[7])),
                    "forward_posts_num": int(float(line[8])),
                    "report_posts_num": int(float(line[9])),
                    "last_signin_time": int(float(line[10]))
                }
                session.run(query, parameters)


def import_post_nodes(file_path):
    #file_path = 'dm_dynamic_profile_10.csv'
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with driver.session() as session:
            for line in reader:
                query = """
                CREATE (:SuperfansPost {
                    # 帖子唯一标识，将字符串转换为整数类型
                    pid: $pid,
                    # 帖子发布的时间戳，将字符串转换为整数类型
                    publish_time: $publish_time,
                    # 帖子关联的明星列表，将 CSV 中的字符串按逗号分割为列表
                    related_stars_list: $related_stars_list,
                    # 帖子获得的点赞数量，将字符串转换为整数类型
                    liked_num: $liked_num,
                    # 帖子获得的评论数量，将字符串转换为整数类型
                    commented_num: $commented_num,
                    # 帖子被转发的数量，将字符串转换为整数类型
                    forwarded_num: $forwarded_num,
                    # 帖子的图片或视频链接，为字符串类型
                    iv_url: $iv_url,
                    # 帖子的文本内容，为字符串类型
                })
                """
                parameters = {
                    "pid": int(float(line[0])),
                    "publish_time": int(float(line[1])),
                    "related_stars_list": line[2].split(","),
                    "liked_num": int(float(line[3])),
                    "commented_num": int(float(line[4])),
                    "forwarded_num": int(float(line[5])),
                    "iv_url": line[6],
                    "text_info": line[7]
                }
                session.run(query, parameters)


import datetime
from dateutil.relativedelta import relativedelta
def import_post_nodes_A(file_path):
    #file_path = 'dm_dynamic_profile_10.csv'
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with driver.session() as session:
            for line in reader:
                query = """
              CREATE (:SuperfansPost_A {
                            // 帖子唯一标识，将字符串转换为整数类型
                            pid: $pid,
                            // 帖子发布的时间戳，将字符串转换为整数类型
                            publish_time: $publish_time,
                            // 帖子关联的明星列表，将 CSV 中的字符串按逗号分割为列表
                            related_stars_list: $related_stars_list,
                            // 帖子获得的点赞数量，将字符串转换为整数类型
                            liked_num: $liked_num,
                            // 帖子获得的评论数量，将字符串转换为整数类型
                            commented_num: $commented_num,
                            // 帖子被转发的数量，将字符串转换为整数类型
                            forwarded_num: $forwarded_num,
                            // 帖子的图片或视频链接，为字符串类型
                            iv_url: $iv_url,
                            // 帖子的文本内容，为字符串类型
                            text_info: $text_info
                        })
                """
                parameters = {
                    "pid": int(float(line[0])),
                    "publish_time": enw_timestamp(int(float(line[1]))),
                    "related_stars_list": line[2].split(","),
                    "liked_num": int(float(line[3])),
                    "commented_num": int(float(line[4])),
                    "forwarded_num": int(float(line[5])),
                    "iv_url": line[6],
                    "text_info": line[7]
                }
                session.run(query, parameters)

def enw_timestamp(timestamp):
    # 将时间戳转换为 datetime 对象
    dt = datetime.datetime.fromtimestamp(timestamp)

    # 减去一个月
    new_dt = dt - relativedelta(months=1)
    # 将新的 datetime 对象转换回时间戳
    new_timestamp = new_dt.timestamp()
    return new_timestamp


if __name__ == "__main__":
        import_user_nodes("D:/pythonwork\django-uwsgi/recomm/import/dm_user_profile_3000.csv")
        import_post_nodes("D:/pythonwork\django-uwsgi/recomm/import/dm_dynamic_profile_10_3000.csv")
        import_post_nodes_A("D:/pythonwork\django-uwsgi/recomm/import/dm_dynamic_profile_10_3000.csv")


