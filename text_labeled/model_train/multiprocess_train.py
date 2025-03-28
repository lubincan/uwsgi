import time
# 用于开启多个进程
import subprocess
# 使用psutil进行资源监控，主要获取cpu与内存占用情况。
import psutil

# 设定CPU与内存的正常和危险占用阈值
CPU_NOR_LIMIT = MEM_NOR_LIMIT = 55
CPU_DAN_LIMIT = MEM_DAN_LIMIT = 95

# 模型训练脚本列表
model_train_list = ["python movie_model_train.py", "python beauty_model_train.py", "python star_model_train.py", "python fashion_model_train.py"]

# 创建subp的列表容器，用于装载子进程
subp = []


def detect_cpu_mem():
    """检测CPU和内存占用率"""
    print("进行mem和cpu检测:")
    # 内存检测
    mem = psutil.virtual_memory().percent
    # psutil检测cpu时间隔至少3s以上
    cpu = psutil.cpu_percent(3)
    print("当前内存占用率:" + str(mem) + "%")
    print("当前cpu占用率:" + str(cpu) + "%")
    return mem, cpu


def single_model_train(model):
    """开启单个模型的训练"""
    p = subprocess.Popen(model, shell=True)
    # 等待3秒预估模型进入训练状态，即资源占用趋于稳定。
    time.sleep(3)
    # 进行资源检测
    mem, cpu = detect_cpu_mem()
    # 内存和CPU同时小于正常负载值，则任其继续运行，并装入列表
    if mem < MEM_NOR_LIMIT and cpu < CPU_NOR_LIMIT:
        subp.append(p)
        print("该模型进入正常训练过程, 并可以开启下一个模型训练!")

    else:
        # 判断是否大于危险负载值，若大于，将kill该进程，
        # 否则等待该进程结束，再进行其他训练任务。
        if mem > MEM_DAN_LIMIT or cpu > CPU_DAN_LIMIT:
            p.kill()
            print("该模型没有进入正常训练过程!")
        else:
            p.wait()
            print("该模型进入正常训练过程, 但不要开启下一个模型训练!")


def start_multiprocess_train():
    """开启多进程训练"""
    print("启动多模型训练:")
    # 遍历启动模型的命令，准备循环开启训练进程
    for i, model in enumerate(model_train_list):
        # 启动模型训练
        print("________________")
        print("正在启动第" + str(i + 1) + "个模型:")
        single_model_train(model)

    else:
        # 所有装入列表的进程都会等待其自然结束后才会停止该函数所在的进程
        print("正在等待所有模型训练结束!")
        list(map(lambda x: x.wait(), subp))
        print("完成！")


if __name__ == "__main__":
    """
    detect_cpu_mem()
    model = model_train_list[0]
    single_model_train(model)
    """
    detect_cpu_mem()
    start_multiprocess_train()
