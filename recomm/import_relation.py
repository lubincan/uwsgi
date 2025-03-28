import pandas as pd
#from neo4j.v1 import GraphDatabase
from neo4j import GraphDatabase

NEO4J_CONFIG = dict({
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "password"),
    "encrypted": False
})
driver = GraphDatabase.driver(**NEO4J_CONFIG)


def write_relationship(relationship_csv_path, relationship_type):
    """写入用户ID与帖子ID节点之间的关系和类型
    :param relationship_csv_path: 关系类型文件路径
    :param relationship_type: 关系类型名称
    :return:
    """
    # 读取关系文件，uid, pid
    uid_and_pid = pd.read_csv(relationship_csv_path, header=None).values

    # driver进行写入
    with driver.session() as session:

        def wr(uid, pid):
            # 找到用户ID
            record = session.run("MATCH(a:SuperfansUser{uid:%s}) return a.uid" % uid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return
            # 找到帖子ID
            record = session.run("MATCH(a:SuperfansPost{pid:%s}) return a.pid" % pid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return

            # 运行写入关系类型
            # uid, pid, 'publish'等名称
            session.run("MATCH(a:SuperfansUser{uid:%s}) MATCH(b:SuperfansPost{uid:%s}) with a,b MERGE(a)-[r:%s]-(b)" % (uid, pid, relationship_type))
        print(uid_and_pid)

        list(map(lambda x: wr(x[0], x[1]), uid_and_pid))


import numpy as np
def write_relationship_new(user_path, postPath, relationship_type):
    """写入用户ID与帖子ID节点之间的关系和类型
    :param relationship_csv_path: 关系类型文件路径
    :param relationship_type: 关系类型名称
    :return:
    """
    # 读取关系文件，uid
    users = pd.read_csv(user_path, header=None).values[:,0]
    fansPosts = pd.read_csv(postPath, header=None).values[:, 0]
    users_index= np.random.choice(len(users), len(users))
    relationship = np.column_stack((users[users_index], fansPosts))
    # driver进行写入
    with driver.session() as session:

        def wr(uid, pid):
            #print(uid,'--------------->',pid)
            # 找到用户ID
            record = session.run("MATCH(a:SuperfansUser{uid:%s}) return a.uid" % uid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return
            # 找到帖子ID
            record = session.run("MATCH(a:SuperfansPost{pid:%s}) return a.pid" % pid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return

            # 运行写入关系类型
            # uid, pid, 'publish'等名称
            session.run("MATCH(a:SuperfansUser{uid:%s}) MATCH(b:SuperfansPost{pid:%s}) with a,b MERGE(a)-[r:%s]-(b)" % (uid, pid, relationship_type))


        list(map(lambda x: wr(x[0], int(float(x[1]))), relationship))



if __name__ == '__main__':

    relationship_type = "like"
    #relationship_csv_path = "/var/lib/neo4j/import/recommend_like_operation_3000.csv"
    print("---------------------------------->",relationship_type)
    relationship_csv_path = "import/recommend_like_operation_3000.csv"
    #write_relationship(relationship_csv_path, relationship_type)
    write_relationship_new("import/dm_user_profile_3000.csv","import/dm_dynamic_profile_10_3000.csv",relationship_type)


    relationship_type = "comment"
    print("---------------------------------->", relationship_type)
    #relationship_csv_path = "/var/lib/neo4j/import/recommend_comment_operation_3000.csv"
    relationship_csv_path = "import/recommend_comment_operation_3000.csv"
    #write_relationship(relationship_csv_path, relationship_type)
    write_relationship_new("import/dm_user_profile_3000.csv","import/dm_dynamic_profile_10_3000.csv",relationship_type)


    relationship_type = "share"
    print("---------------------------------->", relationship_type)
    #relationship_csv_path = "/var/lib/neo4j/import/recommend_share_operation_3000.csv"
    relationship_csv_path = "import/recommend_share_operation_3000.csv"
    #write_relationship(relationship_csv_path, relationship_type)
    write_relationship_new("import/dm_user_profile_3000.csv", "import/dm_dynamic_profile_10_3000.csv",
                           relationship_type)


    relationship_type = "report"
    print("---------------------------------->", relationship_type)
    #relationship_csv_path = "/var/lib/neo4j/import/recommend_report_operation_3000.csv"
    relationship_csv_path = "import/recommend_report_operation_3000.csv"
    #write_relationship(relationship_csv_path, relationship_type)
    write_relationship_new("import/dm_user_profile_3000.csv", "import/dm_dynamic_profile_10_3000.csv",
                           relationship_type)


    # # 指定文件名称
    #relationship_csv_path = "/var/lib/neo4j/import/recommend_post_operation_3000.csv"
    relationship_csv_path = "import/recommend_post_operation_3000.csv"
    # 指定关系类型
    relationship_type = 'publish'
    print("---------------------------------->", relationship_type)
    #write_relationship(relationship_csv_path, relationship_type)
    #write_relationship_new("import/dm_user_profile_3000.csv", "import/dm_dynamic_profile_10_3000.csv", relationship_type)




