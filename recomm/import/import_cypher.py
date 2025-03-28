import pandas as pd
import fileinput
#from neo4j.v1 import GraphDatabase
from neo4j import GraphDatabase
NEO4J_CONFIG = dict({
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "password"),
    #'encrpted': False,
    "encrypted": False
})


driver = GraphDatabase.driver(**NEO4J_CONFIG)



def wirte_relationship(relationship_csv_path, relationship_type):
    #uid_and_pid = list(map(lambda x: x.strip(), fileinput.FileInput(relationship_csv_path)))
    uid_and_pid = pd.read_csv(relationship_csv_path, header=None).values
    #print(uid_and_pid)
    with driver.session() as session:
    

        def wr(uid, pid):
            print(uid, pid)
            #print("MATCH(a:SuperfansUser{uid:%s}) return a.uid" %uid)
            record = session.run("MATCH(a:SuperfansUser{uid:%s}) return a.uid" %uid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return
            #print("MATCH(a:SuperfansPost{pid:%s}) return a.pid" %pid)
            record = session.run("MATCH(a:SuperfansPost{pid:%s}) return a.pid" %pid)
            result = list(map(lambda x: x[0], record))
            if not result:
                return

            #print("MATCH(a:SuperfansUser{uid:%s})-[r:%s]-(b:SuperfansPost{pid:%s}) SET r.num=r.num+1" %(uid, relationship_type, pid))
            record = session.run("MATCH(a:SuperfansUser{uid:%s})-[r:%s]-(b:SuperfansPost{pid:%s}) SET r.num=r.num+1 return r.num" %(uid, relationship_type, pid))
            result = list(map(lambda x: x[0], record))
            if not result:
                #print("MATCH(a:SuperfansUser{uid:%s}) MATCH(b:SuperfansPost{pid:%s}) with a, b MERGE(a)-[r:%s]-(b) SET r.num=1" %(uid, pid, relationship_type))
                session.run("MATCH(a:SuperfansUser{uid:%s}) MATCH(b:SuperfansPost{pid:%s}) with a, b MERGE(a)-[r:%s]-(b) SET r.num=1" %(uid, pid, relationship_type))
        list(map(lambda x: wr(x[0], x[1]), uid_and_pid))



if __name__ == "__main__":

    relationship_type = "publish"
    relationship_csv_path = "./recommend_post_operation_3000.csv"
    wirte_relationship(relationship_csv_path, relationship_type)

    relationship_type = "like"
    relationship_csv_path = "./recommend_like_operation_3000.csv"
    wirte_relationship(relationship_csv_path, relationship_type)

    relationship_type = "comment"
    relationship_csv_path = "./recommend_comment_operation_3000.csv"
    wirte_relationship(relationship_csv_path, relationship_type)

    relationship_type = "share"
    relationship_csv_path = "./recommend_share_operation_3000.csv"
    wirte_relationship(relationship_csv_path, relationship_type)

    relationship_type = "report"
    relationship_csv_path = "./recommend_report_operation_3000.csv"
    wirte_relationship(relationship_csv_path, relationship_type)
