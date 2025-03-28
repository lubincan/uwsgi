# 因为我们需要导入settings.py中的配置信息
# 所以需要将上一级路径导入到系统路径中
import sys
sys.path.append('../')
from  text_labeled.settings import LABEL_STRUCTURE
from config import NEO4J_CONFIG,LABEL_STRUCTURE
from neo4j import GraphDatabase

import json
def create_label_node_and_rel():
    """该函数用于创建标签树的节点和边"""
    _driver = GraphDatabase.driver(**NEO4J_CONFIG)
    with _driver.session() as session:
        # 删除所有Label节点以及相关联的边
        cypher = "MATCH(a:Label) DETACH DELETE a"
        session.run(cypher)

        def _create_node_rel(l: dict):
            """根据标签树结构中的每一个字典去创建节点和关系"""
            if not l: return
            # 遍历字典中的k,v即父子标签
            for k, v in l.items():
                # MERGE一个父标签节点
                cypher = "MERGE(a:Label{title: %r})" % (k)
                session.run(cypher)

                def __c(word):
                    """用于创建子标签节点以及与父标签之间的关系"""
                    cypher = "CREATE(a:Label{title: %r}) \
                              SET a.name=%r WITH a \
                              MATCH(b: Label{title: %r}) \
                              MERGE(b)-[r:Contain]-(a)" % (word, word, k)
                    session.run(cypher)

                # 遍历子标签列表
                list(map(__c, v))

        # 遍历标签树列表
        list(map(_create_node_rel, LABEL_STRUCTURE))
        get_all_node_rel()




def get_all_node_rel():
    _driver = GraphDatabase.driver(**NEO4J_CONFIG)
    with _driver.session() as session:
        cypher = "MATCH p=()-[r:Contain]->() RETURN p "
        result = session.run(cypher)

        # 定义一个函数来将节点转换为字典
        def node_to_dict(node):
            node_dict = dict(node)
            node_dict["id"] = node.element_id
            node_dict["labels"] = list(node.labels)
            return node_dict

            # 定义一个函数来将关系转换为字典

        def relationship_to_dict(relationship):
            rel_dict = dict(relationship)
            rel_dict["id"] = relationship.element_id
            rel_dict["type"] = relationship.type
            rel_dict["start_node"] = relationship.start_node.element_id
            rel_dict["end_node"] = relationship.end_node.element_id
            return rel_dict

            # 定义一个函数来将路径转换为字典

        def path_to_dict(path):
            path_dict = {
                "nodes": [node_to_dict(node) for node in path.nodes],
                "relationships": [relationship_to_dict(rel) for rel in path.relationships]
            }
            return path_dict

        results_list = []
        for record in result:
            path = record["p"]
            path_dict = path_to_dict(path)
            results_list.append(path_dict)

    # 将结果列表转换为 JSON 字符串
    json_data = json.dumps(results_list, indent=4)
    print('所有标签关系', json_data)
    return results_list

import os
from config import NEO4J_CONFIG,LABEL_STRUCTURE
from neo4j import GraphDatabase
import random
csv_path = "./labels"
def create_vocabulary_node_and_rel():
    """该函数用于创建词汇节点和关系"""
    _driver = GraphDatabase.driver(**NEO4J_CONFIG)
    with _driver.session() as session:
        #删除所有词汇节点及其相关的边
        cypher = "MATCH(a:Vocabulary) DETACH DELETE a"
        session.run(cypher)

        def _create_v_and_r(csv):
            """读取单个csv文件,并写入数据库创建节点并与对应的标签建立关系"""
            path = os.path.join(csv_path,csv)
            # 使用fileinput的FileInput方法从持久化文件中读取数据,
            # 并进行strip()操作去掉两侧空白符, 再通过set来去重.
            word_set = set()
            try:
                with open(path, 'r', encoding='UTF-8') as file:
                    for line in file:
                        word = line.strip()
                        if word:
                            word_set.add(word)

            except FileNotFoundError:
                print(f"未找到文件: {path}")
            word_list = list(set(map(lambda x: x.strip(), word_set)))

            def __create_node(word):
                """创建csv中单个词汇的节点和关系"""
                # 定义词汇的初始化权重,即词汇属于某个标签的初始概率，
                # 因为词汇本身来自该类型文章，因此初始概率定义在0.5-1之间的随机数
                weight = round(random.uniform(0.5, 1), 3)
                # 使用cypher语句创建词汇节点,然后匹配这个csv文件名字至后四位即类别名，
                # 在两者之间MERGE一条有权重的边
                # 使用cypher语句创建词汇节点,然后匹配这个csv文件名字至后四位即类别名，
                # 在两者之间MERGE一条有权重的边
                cypher = "CREATE(a:Vocabulary{name:%r}) WITH a \
                                         MATCH(b:Label{title:%r}) \
                                         MERGE(a)-[r:Related{weight:%f}]-(b)" % (word, csv[:-4], weight)
                session.run(cypher)
            # 遍历词汇列表
            list(map(__create_node,word_list))
        # 遍历标签列表
        label_list = os.listdir(csv_path)
        print(label_list)
        list(map(_create_v_and_r,label_list))

if __name__ == "__main__":
    #create_label_node_and_rel()
    create_vocabulary_node_and_rel()