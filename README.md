# uwsgi —— 泛娱乐推荐系统

> 从零搭建一个工业级推荐系统：图数据库召回 + 深度学习排序 + 知识图谱标签识别

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-brightgreen.svg)](https://neo4j.com/)
[![Redis](https://img.shields.io/badge/Redis-Cache-red.svg)](https://redis.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 为什么做这个项目？

我发现市面上大多数开源项目要么过于简单（只有协同过滤），要么过于复杂（缺少详细文档）。**真正能帮助初学者理解工业级推荐系统全貌的项目少之又少**。

于是，我决定把自己在学习推荐系统过程中的实践成果完整开源出来。这个项目可能不是最完美的，但**每一个模块都是我一行一行代码调试出来的**，包含了我踩过的坑和思考过程。

希望这个项目能帮助同样在学习推荐系统的你，**少走一些弯路**。

## 🎯 项目简介

**uwsgi** 是一个基于 **Django + Neo4j + Redis + TensorFlow/Keras** 的泛娱乐内容推荐与文本标签识别系统。系统包含两大核心能力：

1. **泛娱乐内容推荐**：以图数据库为召回核心，融合多种召回策略 + 规则过滤 + 缓存排序，实现「千人千面」的帖子推荐。
2. **文本标签识别**：基于深度学习（Embedding + n-gram + 神经网络）对文章文本进行多标签分类，并通过知识图谱对标签进行歧义消解与归一化。

### 🚀 你能学到什么？

- ✅ 多路召回策略的设计与实现
- ✅ 图数据库在推荐系统中的应用
- ✅ Wide & Deep 模型的实战应用
- ✅ 知识图谱的构建与使用
- ✅ 文本多标签分类的完整流程
- ✅ Redis 缓存在推荐系统中的作用
- ✅ 工业级代码的组织结构

---

## 📑 目录

- 一技术架构
- 核心模块详解
- 环境准备
- 快速上手
- 一起学习

---

## 一、技术架构

### 1.1 技术栈总览

| 层次 | 技术选型 | 用途 | 版本要求 |
|------|----------|------|----------|
| Web 框架 | Django + Django REST Framework | 提供 HTTP API 服务 | 4.2+ |
| 图数据库 | Neo4j | 标签树、词汇-标签关系、用户-帖子行为关系存储 | 4.x+ |
| 缓存 | Redis | IP→UID 映射、推荐结果金字塔缓存 | 5.x+ |
| 深度学习 | TensorFlow / Keras | 文本分类模型训练与预测 | 2.10 |
| 分词 | jieba | 中文分词、词性标注、自定义/停用词典 | 0.42+ |
| 数据处理 | pandas / numpy / scikit-learn | 样本构建、特征处理、数据分析 | 最新稳定版 |
| 图像去重 | OpenCV (cv2) | 帖子图片指纹（DCT + 汉明距离）去重 | 4.x+ |
| 环境管理 | conda | 依赖与运行环境管理 | Python 3.9 |

### 1.2 项目目录结构

```
uwsgi/
├── manage.py                    # Django 命令行入口
├── db.sqlite3                   # Django 默认数据库（未使用业务表）
├── environment.txt              # conda 环境导出文件
├── README.md                    # 项目说明文档（本文件）
├── requirements.txt             # Python 依赖清单
├── uwsgi/                       # Django 项目配置
│   ├── settings.py              # 项目配置（含 CORS）
│   ├── urls.py                  # 根路由配置
│   ├── wsgi.py                  # WSGI 入口
│   └── asgi.py                  # ASGI 入口
├── api/                         # 对外 API 应用
│   ├── urls.py                  # API 路由配置
│   └── views.py                 # 视图函数（请求处理）
├── recomm/                      # 推荐系统模块
│   ├── api.py                   # 召回/过滤/缓存/行为写入
│   ├── import_relation.py       # 用户-帖子关系导入
│   ├── import_node.sh           # Cypher 节点导入脚本
│   ├── import/                  # 数据导入脚本与 CSV
│   │   ├── node_import_cypher.py    # Neo4j 节点导入脚本
│   │   ├── import_cypher.py         # Cypher 导入工具
│   │   ├── aTOb.py                  # 关系转换工具
│   │   └── *.csv                    # 用户画像/帖子/行为数据
│   └── model/                   # Wide & Deep 排序模型
│       └── trainer/
│           ├── model.py         # 特征列与模型定义
│           └── task.py          # 训练任务入口
└── text_labeled/                # 文本标签识别模块
    ├── api.py                   # 分词、图谱匹配、标签归一化流程
    ├── settings.py              # 标签树结构定义 (LABEL_STRUCTURE)
    ├── userdict.txt             # jieba 用户自定义词典
    ├── stopdict.txt             # jieba 停用词典
    ├── create_graph/            # 知识图谱构建
    │   ├── build.py             # 标签节点/词汇节点与关系创建
    │   ├── get_vocab.py         # 从文章抽取名词词汇
    │   └── beauty|fashion|movie|star/   # 原始文章语料
    └── model_train/             # 文本分类模型训练与预测
        ├── model_all_train.py   # 特征工程 + 模型训练
        ├── multiprocess_train.py    # 多进程并行训练
        ├── multithread_predict.py   # 多线程预测 / H5 & PB 模型服务
        ├── get_sample.py        # 正负样本构建
        ├── data_analysis.py     # 样本分布分析
        ├── model_config.json    # 模型微服务配置
        └── beauty|fashion|movie|star_*_train.py  # 各标签训练入口
```

### 1.3 架构分层

```
┌─────────────────────────────────────────────────────┐
│                    前端 / 客户端                       │
│              (Web/Mobile/Third-party)                │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (REST API)
┌──────────────────────▼──────────────────────────────┐
│              Django API 层 (api/views.py)            │
│   请求解析 → 业务编排 → 结果序列化返回                 │
└───────────┬──────────────────────────┬───────────────┘
            │                          │
   ┌────────▼─────────┐      ┌─────────▼─────────────┐
   │  recomm/api.py   │      │  text_labeled/api.py  │
   │  推荐服务模块     │      │  文本标签识别服务        │
   └───┬──────────┬───┘      └────┬────────────┬─────┘
       │          │               │            │
  ┌────▼───┐  ┌───▼────┐    ┌─────▼────┐  ┌────▼──────┐
  │ Neo4j  │  │ Redis  │    │  Neo4j   │  │  Keras   │
  │ 图存储  │  │ 缓存   │    │ 标签/词  │  │ 分类模型  │
  └────────┘  └────────┘    └──────────┘  └───────────┘
```

---

## 二、核心模块详解

### 2.1 泛娱乐推荐模块（`recomm/`）

#### 🔄 推荐整体流程

推荐流程（`recomm/api.py` 中的 `_get_recomm`）遵循「**召回 → 合并排序 → 缓存 → 过滤**」的经典推荐架构：

```
用户请求 → 多路召回 → 金字塔合并排序 → Redis缓存去重 → 规则过滤 → 返回结果
```

#### 📊 多路召回策略

1. **`_get_hot()` 热门召回**
   - 计算公式：`hot_score = 2*评论数 + 点赞数 + 3*转发数`
   - 按热度分数排序，取 Top 100
   - 适用于冷启动场景

2. **`get_last()` 时间召回**
   - 按发布时间倒序排列
   - 取最新发布的 Top 100 帖子
   - 保证内容时效性

3. **`get_v()` 速度召回**
   - 计算单位时间内热度增速
   - 识别快速上升的热门内容
   - 捕捉热点趋势

4. **`get_r(uid)` 协同过滤召回**
   - 基于用户的二度关系协同召回
   - 路径：`用户 → 帖子 → 用户 → 帖子`
   - 利用图数据库的关系传递性

5. **`get_random()` 随机召回**
   - 运营自定义随机召回
   - 增加推荐多样性
   - 避免信息茧房

#### 🏗️ 金字塔合并排序（`pyramid_array`）

- 统计每个 `pid`（帖子ID）在多少路召回结果中出现
- 出现次数越多，权重越高
- 按权重降序输出最终排序结果
- 融合多路召回优势

#### 💾 缓存写入（`j_data_write`）

- 将排序结果写入 Redis（键：`j_uid`）
- 对比历史缓存（键：`r_uid`）进行去重
- 实现「翻页不重复」的推荐体验
- 优化响应速度

#### 🔍 规则过滤（`rfilter`）

- 基于 OpenCV 的图像指纹技术
- DCT（离散余弦变换）+ 汉明距离算法
- 对同批次相似图片进行去重
- 提升内容多样性

#### 🧠 排序模型（`recomm/model/trainer/`）

- 使用 TensorFlow `DNNLinearCombinedClassifier`
- Wide & Deep 混合结构
- 融合特征：
  - 用户画像（关注明星、兴趣偏好等）
  - 帖子特征（内容特征、作者特征等）
- 预测目标：点击率（`islike`）

#### 📝 行为写入（`write_to_neo4j` / `cancel_to_neo4j`）

- 实时记录用户行为
- 支持：点赞、评论、转发
- 写入图数据库形成行为关系
- 为协同过滤提供数据基础

### 2.2 文本标签识别模块（`text_labeled/`）

#### 📋 完整识别流程

文本标签识别流程（`text_labeled/api.py`）：

```
输入文本 → 中文分词 → 图谱匹配 → 权重更新/歧义消解 → 概率归并 → 归一化 → 输出标签
```

#### 1️⃣ 文本预处理（`handle_cn_text`）

- 使用 jieba 进行中文分词
- 加载自定义词典 `userdict.txt`
- 加载停用词典 `stopdict.txt`
- 过滤单字与停用词
- 保留有意义的词汇

#### 2️⃣ 图谱匹配（`get_index_map_label`）

- 在 Neo4j 中执行图查询
- 匹配路径：`Vocabulary` 词汇节点 → `Related` 边 → `Label` 标签节点
- 返回词汇索引与对应的「标签-权重」列表
- 利用图结构的关联性

#### 3️⃣ 权重更新 / 歧义消解（`weight_update`）

- 处理一词多义问题
- 当一个词汇对应多个标签时（存在歧义）
- 调用深度学习模型重新预测概率
- 基于上下文进行消歧

#### 4️⃣ 概率归并（`control_increase`）

- 使用 pandas 进行数据处理
- 对同一标签的概率求和
- 得到标签级总得分
- 聚合多个词汇的贡献

#### 5️⃣ 归一化与父标签检索（`father_label_and_normalized`）

- 使用 sigmoid 函数归一化
- 得分映射到 `[0,1]` 区间
- 在标签树中向上检索父标签
- 通过 `Contain` 关系追溯
- 输出最终标签及关联标签

#### 🌳 标签树结构

定义在 `settings.py` 的 `LABEL_STRUCTURE`：

```
泛娱乐
├── 明星
├── 时尚
├── 游戏
│   ├── LOL
│   ├── 王者农药
│   └── 吃鸡
├── 影视
│   ├── 喜剧
│   ├── 综艺
│   ├── 科幻
│   └── 恐怖
├── 音乐
│   ├── 摇滚乐
│   ├── 民谣
│   ├── Rap
│   └── 流行乐
└── 美妆
```

### 2.3 文本分类模型（`text_labeled/model_train/`）

#### 🧬 模型架构

- 结构：`Embedding + GlobalAveragePooling1D + Dense(sigmoid)`
- 类型：二分类模型
- 策略：对每个标签类别训练独立模型

#### 🔧 特征工程

1. **词汇映射**：使用 `Tokenizer` 建立词汇表
2. **序列对齐**：使用 `pad_sequences` 截断对齐
3. **特征增强**：加入 n-gram 特征（2-gram）
4. **向量化**：转换为模型输入格式

#### 🚀 模型训练

- **训练脚本**：`model_all_train.py`
- **优化器**：Adam
- **损失函数**：二分类交叉熵（binary_crossentropy）
- **正则化**：早停机制（Early Stopping）
- **验证策略**：训练集/验证集划分

#### ⚡ 并行训练

- **脚本**：`multiprocess_train.py`
- **实现方式**：通过 subprocess 开启多进程
- **资源调度**：根据 CPU/内存占用动态调度
- **效率提升**：同时训练多个标签模型

#### 🎯 模型预测

- **H5 格式**：
  - `predict_test_h5` 直接本地加载模型预测
  - 适合开发测试环境
  
- **PB 格式**：
  - `to_savedmodel` 将 H5 转为 TensorFlow SavedModel
  - 配合 TensorFlow Serving 使用
  - 多线程调用（`request_model_serve_thread`）
  - 适合生产环境部署

---

## 三、环境准备

### 3.1 依赖安装

#### 方式一：使用 conda 环境文件（推荐）

项目使用 conda 管理环境（Python 3.9），可通过 `environment.txt` 一键重建环境：

```bash
# 创建环境
conda create --name uwsgi --file environment.txt

# 激活环境
conda activate uwsgi
```

#### 方式二：手动安装依赖

```bash
# 安装 Django 相关
pip install django==4.2 djangorestframework django-cors-headers django-filter

# 安装数据库驱动
pip install neo4j redis

# 安装数据处理库
pip install jieba pandas numpy scikit-learn matplotlib

# 安装深度学习框架
pip install tensorflow==2.10 keras==2.10

# 安装图像处理库
pip install opencv-python
```

### 3.2 外部服务启动

系统依赖以下外部服务，需确保已启动并可访问：

| 服务 | 默认地址 | 默认账号/密码 | 说明 |
|------|----------|---------------|------|
| Neo4j | `bolt://127.0.0.1:7687` | `neo4j/password` | 图数据库，存储关系数据 |
| Redis | `127.0.0.1:6379` | 无密码 | 缓存服务，存储推荐结果 |

> **⚠️ 重要提示**：数据库连接配置见 `recomm/api.py`（`NEO4J_CONFIG`、`REDIS_CONFIG`）与 `text_labeled` 相关模块。若您的配置不同，请同步修改对应文件。

### 3.3 环境验证

```bash
# 验证 Python 版本
python --version  # 应显示 Python 3.9.x

# 验证 Django 安装
python -c "import django; print(django.get_version())"

# 验证 TensorFlow 安装
python -c "import tensorflow as tf; print(tf.__version__)"

# 验证 Neo4j 连接
python -c "from neo4j import GraphDatabase; print('Neo4j driver OK')"

# 验证 Redis 连接
python -c "import redis; r = redis.Redis(); print(r.ping())"
```

---

## 四、快速上手

### 4.1 启动 Django 服务

```bash
# 进入项目目录
cd uwsgi

# 启动开发服务器
python manage.py runserver 0.0.0.0:8000

# 后台运行（可选）
nohup python manage.py runserver 0.0.0.0:8000 &
```

服务启动后即可通过 API 访问：`http://127.0.0.1:8000`

### 4.2 推荐系统数据准备

#### 步骤 1：导入节点数据

```bash
# 方法一：使用 Python 脚本（推荐）
python recomm/import/node_import_cypher.py

# 方法二：使用 Cypher 脚本
# 需要在 Neo4j 环境中执行
bash recomm/import_node.sh
```

导入内容：
- 用户节点（3000 个用户）
- 帖子节点（动态内容）
- 用户画像属性

#### 步骤 2：导入关系数据

```bash
python recomm/import_relation.py
```

导入关系类型：
- 发布关系（用户-帖子）
- 点赞关系
- 评论关系
- 转发关系
- 举报关系

### 4.3 文本标签图谱构建

#### 步骤 1：创建标签树节点与关系

```bash
# 命令行方式
python text_labeled/create_graph/build.py

# 或通过 API 方式
curl http://127.0.0.1:8000/api/create/label/rel/
```

创建内容：
- 标签节点（泛娱乐、明星、时尚等）
- 标签层级关系（Contain 关系）

#### 步骤 2：抽取词汇并创建词汇节点

```bash
# 命令行方式
python text_labeled/create_graph/get_vocab.py

# 或通过 API 方式
curl http://127.0.0.1:8000/api/label/create/words/
curl http://127.0.0.1:8000/api/vocabulary/node/rel/
```

处理流程：
- 从文章语料中抽取名词词汇
- 创建词汇节点
- 建立词汇-标签关系

### 4.4 文本分类模型训练

#### 步骤 1：构建正负样本

```bash
python text_labeled/model_train/get_sample.py
```

生成内容：
- 正样本（标签相关的文章）
- 负样本（标签无关的文章）
- 样本标签标注

#### 步骤 2：训练模型

```bash
# 单标签训练
python text_labeled/model_train/model_all_train.py

# 多进程并行训练（推荐）
python text_labeled/model_train/multiprocess_train.py

# 特定标签训练示例
python text_labeled/model_train/beauty_all_train.py
python text_labeled/model_train/fashion_all_train.py
python text_labeled/model_train/movie_all_train.py
python text_labeled/model_train/star_all_train.py
```

#### 步骤 3：模型预测

```bash
# 本地 H5 格式预测
python text_labeled/model_train/multithread_predict.py

# TensorFlow Serving 预测（需要先转换模型）
python text_labeled/model_train/to_savedmodel.py
```

### 4.5 完整测试流程

```bash
# 1. 测试文本标签识别
curl -X POST http://127.0.0.1:8000/api/get_label \
  -H "Content-Type: application/json" \
  -d '{"text": "今天看了周杰伦的演唱会，太精彩了！"}'

# 2. 测试推荐系统
curl http://127.0.0.1:8000/api/first_show

# 3. 测试分词
curl "http://127.0.0.1:8000/api/cntext?text=今天天气不错"
```

---
## 最后说几句

推荐系统教会我一件事：**从海量候选中找到最相关的那几条，这个能力几乎可以迁移到任何技术方向。** RAG 的检索、搜索排序、信息流广告，骨架都是"召回-排序-过滤"。

所以如果正在往大模型应用方向走，推荐系统是一个很实在的起点。它具体到能写进简历，又通用到能带你去更远的地方。

这个项目我会一直开源，一直免费。

但有些东西，确实放不进 README。

凌晨一点对着 Cypher 报错发呆，搜遍全网找不到一个能用的答案，翻了四页都是同一篇文章的搬运。理论看了不少，一落地就抓瞎。代码能跑通，却说不清为什么这么设计。想找人聊聊，翻遍通讯录，没有一个在做同样的事。

这些时刻，我一个人经历过太多次。

后来慢慢意识到，**技术上的坎，有时候不是卡在代码本身，是卡在"只有你一个人"**。

所以我把这个项目写出来，也维护了一个小圈子，**AI大模型工程社**。里面没有噱头，就是一群在做同一件事的人。分享一些文档里写不下的东西：关键代码的注释版、设计决策的推演过程、面试高频题的拆解思路，还有那些我踩坑时最希望有人提前告诉我的一句话。

有人问得浅，有人答得深，没人觉得谁的问题蠢。因为这个阶段，大家都走过。

**如果你现在也觉得一个人学习大模型应用有点难，可以来看看。不为了别的，就为了下次凌晨卡住的时候或想实现token自由，知道去哪儿问一句。**

👉 [一起聊聊技术](https://share.note.youdao.com/s/3Hc9ju2)

**顺手给个 Star 吧。让我知道，这条路上不止我一个人。** 🚀