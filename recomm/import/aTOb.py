import  csv
import pandas as pd
import numpy as np


def relation():
    uid_and_pid = pd.read_csv("recommend_post_operation_3000.csv", header=None).values
    users = pd.read_csv("dm_user_profile_3000.csv", header=None).values
    fansPosts = pd.read_csv("dm_dynamic_profile_10_3000.csv", header=None).values

    new_uid_pid = np.column_stack((fansPosts[:,0], users[:,0]))
    print(new_uid_pid)

    # 定义两个一维 NumPy 数组
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])

    # 组合次数
    n = 5  # 可根据需要修改组合次数
    random_indices1 = np.random.choice(len(array1), len(array1))

    combined_2d_array = np.column_stack((array1[random_indices1], array2))
    print(random_indices1)
    print("随机组合后的二维数组：")
    print(combined_2d_array)


from itertools import chain
def pyramid_array(all_data):
    result = []
    for pid in set(chain(*all_data)):
        v = 0
        for list_ in all_data:
            if pid in list_:
                v += 1
        result.append([pid, v])

    result.sort(key=lambda x: x[1])
    print('result--->',result)
    return list(map(lambda x: x[0], result))[::-1]

if __name__ =='__main__':
    all_data = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]

    result = pyramid_array(all_data)
    print(result)