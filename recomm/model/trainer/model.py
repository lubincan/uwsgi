import tensorflow as tf

CSV_COLUMNS = [
'like_posts_num', 'forward_posts_num', 'comment_posts_num', 'publish_posts_num', 'hot_score',
    'commented_num', 'forwarded_num', 'liked_num', 'publish_time', 'follow_star_1',
    'follow_star_2', 'follow_star_3', 'follow_star_4', 'follow_star_5', 'related_star_1',
    'related_star_2', 'related_star_3', 'related_star_4', 'related_star_5', 'islike'
]
CSV_COLUMN_DEFAULTS = [[''], [0], [0], [0], [0],
                       [0], [0], [0], [0], [0],
                       [0], [0], [0], [0], [0],
                       [0], [0], [0], [0], ['']]

LABEL_COLUMN = 'islike'
LABELS = ['1', '0']
STAR_ID_LIST = list(map(lambda x: x, range(0,500)))

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('like_posts_num'),
    tf.feature_column.numeric_column('forward_posts_num'),
    tf.feature_column.numeric_column('comment_posts_num'),
    tf.feature_column.numeric_column('publish_posts_num'),
    tf.feature_column.numeric_column('commented_num'),
    tf.feature_column.numeric_column('forwarded_num'),
    tf.feature_column.numeric_column('liked_num'),
    tf.feature_column.numeric_column('publish_time'),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_1', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_2', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_3', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_4', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_5', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_1', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_2', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_3', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_4', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_5', STAR_ID_LIST)
]


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
    """输入数据的读取函数
    :param filenames: 文件名
    :param num_epochs: 迭代所有样本次数
    :param shuffle: 是否打乱
    :param skip_header_lines: 跳过头部
    :param batch_size: 每批次处理大小
    :return:
    """
    def _decode_csv(line):
        """对于CSV文件每一行做处理
        :param line: 一行结果
        :return: 特征字典
        """
        # 扩充维度到二维
        row_columns = tf.expand_dims(line, -1)
        # decode_csv指定类型和默认值
        columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)

        # 封装成字典，删除目标值
        features = dcit(zip(CSV_COLUMNS, columns))

        return features

    # dataset
    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(_decode_csv)
    # map repeat shuffle batch
    # 是否打乱顺序
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
    # 迭代次数，每批次处理大小
    features = iterator.get_next()

    def parse_label_column(label_string_tensor):
        table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
        return table.lookup(label_string_tensor)

    return features, parse_label_column(features.pop(LABEL_COLUMN))


def build_estimator(config, embedding_size=8):
    """建立模型输入特征列，以及模型构建
    :param config: 模型训练的配置
    :param embedding_size: 初始输入大小
    :return:
    """

    (like_posts_num, forward_posts_num, comment_posts_num, publish_posts_num, hot_score,
     commented_num, forwarded_num, liked_num, publish_time, follow_star_1,
     follow_star_2, follow_star_3, follow_star_4, follow_star_5, related_star_1,
     related_star_2, related_star_3, related_star_4, related_star_5) = INPUT_COLUMNS

    # 指定wide侧输入的特征
    wide_columns = [
        tf.feature_column.crossed_column([follow_star_1, related_star_1],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_1, related_star_2],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_1, related_star_3],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_1, related_star_4],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_1, related_star_5],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_2, related_star_1],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_2, related_star_2],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_2, related_star_3],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_2, related_star_4],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_2, related_star_5],
                                         hash_bucket_size=int(1e3)),
        tf.feature_column.crossed_column([follow_star_1, related_star_1, follow_star_2],
                                         hash_bucket_size=int(1e4)),
        tf.feature_column.crossed_column([follow_star_2, related_star_1, related_star_2],
                                         hash_bucket_size=int(1e4)),
        tf.feature_column.crossed_column([follow_star_3, related_star_1, related_star_2],
                                         hash_bucket_size=int(1e4)),
        tf.feature_column.crossed_column([follow_star_1, related_star_2, related_star_1],
                                         hash_bucket_size=int(1e4)),
        device_system,
        follow_star_1,
        follow_star_2,
        follow_star_3,
        follow_star_4,
        follow_star_5,
        related_star_1,
        related_star_2,
        related_star_3,
        related_star_4,
        related_star_5
    ]

    # 指定deep侧输入的特征
    # 深度特征比做挖掘特征，针对稀疏+稠密的所有特征, 但由于隐层作用时将考虑大小问题，因此类别特征必须onehot编码才能作为输入
    deep_columns = [
        tf.feature_column.indicator_column(follow_star_1),
        tf.feature_column.indicator_column(follow_star_2),
        tf.feature_column.indicator_column(follow_star_3),
        tf.feature_column.indicator_column(follow_star_4),
        tf.feature_column.indicator_column(follow_star_5),
        tf.feature_column.indicator_column(related_star_1),
        tf.feature_column.indicator_column(related_star_2),
        tf.feature_column.indicator_column(related_star_3),
        tf.feature_column.indicator_column(related_star_4),
        tf.feature_column.indicator_column(related_star_5),
        like_posts_num,
        forward_posts_num,
        comment_posts_num,
        publish_posts_num,
        commented_num,
        forwarded_num,
        liked_num,
        publish_time
    ]

    return tf.estimator.DNNLinearCombinedClassifier(config=config,
                                                    linear_feature_columns=wide_columns,
                                                    dnn_feature_columns=deep_columns,
                                                    dnn_hidden_units=[embedding_size] + [100, 70, 50, 25])


if __name__ == '__main__':
    pass