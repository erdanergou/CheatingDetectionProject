import time
from abc import ABC, abstractmethod

import torch

single_process = True
from queue import Empty

if single_process:
    from queue import Queue
    from threading import Thread, Lock
else:
    from torch.multiprocessing import Queue, Lock
    from torch.multiprocessing import Process as Thread

    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

queueSize = 50

TASK_DATA_OK = 0  # 成功返回数据
TASK_DATA_CLOSE = 1  # 关闭数据
TASK_DATA_IGNORE = 2  # 忽视数据
TASK_DATA_SKIP = 3  # 跳过数据
BALANCE_CEILING_VALUE = 50  # 平衡上限数据


class DictData(object):
    def __init__(self):
        pass


class ModuleBalancer:
    def __init__(self):
        self.max_interval = 0  # 最大间隔
        self.short_stab_interval = self.max_interval  # 短暂尝试间隔
        self.short_stab_module = None  # 短暂尝试模型
        self.lock = Lock()  # 锁
        self.ceiling_interval = 0.1  # 下间隔

    def get_suitable_interval(self, process_interval, module):
        # 得到合适的间隔
        with self.lock:  # 如果没有锁
            if module == self.short_stab_module:
                self.max_interval = (process_interval + self.max_interval) / 2
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            elif process_interval > self.short_stab_interval:
                self.short_stab_module = module
                self.max_interval = process_interval
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            else:
                return max(min(self.max_interval - process_interval, self.ceiling_interval), 0)


class TaskData:
    # 工作数据
    def __init__(self, task_stage, task_flag=TASK_DATA_OK):
        self.data = DictData()  # data为字典格式
        self.task_stage = task_stage  # 任务阶段
        self.task_flag = task_flag  # 任务标志


class TaskStage:
    # 工作阶段
    def __init__(self):
        self.next_module = None  # 下个模型
        self.next_stage = None  # 下个阶段

    def to_next_stage(self, task_data: TaskData):  # 到下一阶段
        self.next_module().put_task_data(task_data)  # 设置下阶段的数据
        task_data.task_stage = self.next_stage  # 下阶段的工作阶段


class BaseModule(ABC):
    # 基础模型
    def __init__(self, balancer=None, skippable=True):
        self.skippable = skippable  # 是否可跳过
        self.ignore_task_data = TaskData(task_stage=None, task_flag=TASK_DATA_IGNORE)  # 忽视工作数据
        self.queue = Queue(maxsize=queueSize)  # 设置队列
        self.balancer: ModuleBalancer = balancer  # 平衡器
        self.process_interval = 0.01  # 进程间隔
        self.process_interval_scale = 1  # 过程等距量表
        print(f'created: {self}')

    @abstractmethod
    def process_data(self, data):
        pass

    @abstractmethod
    def open(self):
        self.running = True  # 设置运行位True

    def _run(self):
        # 运行
        self.running = True
        self.open()  # 打开
        while self.running:  # 如果running为True则一直运行
            task_data = self.product_task_data()  # 产生工作数据
            # 执行条件
            execute_condition = task_data.task_flag == TASK_DATA_OK  # 如果工作数据的标志位是Ok，则设为True
            # 如果执行条件为True或者工作数据的标志位为跳过并且跳过的标志为False
            execute_condition = execute_condition or (task_data.task_flag == TASK_DATA_SKIP and not self.skippable)
            # 执行和执行结果
            start_time = time.time()  # 设置开始时间
            # 执行结果：如果执行条件为True，则为数据预处理后的值，否则为标志位
            execute_result = self.process_data(task_data.data) if execute_condition else task_data.task_flag
            # 处理间隔
            process_interval = min((time.time() - start_time) * self.process_interval_scale, BALANCE_CEILING_VALUE)
            task_stage = task_data.task_stage  # 工作阶段
            if execute_result != TASK_DATA_SKIP:  # 如果执行结果不是跳过
                self.process_interval = process_interval
            else:
                task_data.task_flag = TASK_DATA_SKIP
            if execute_result == TASK_DATA_IGNORE:  # 如果执行结果为忽略
                continue
            else:
                if execute_result == TASK_DATA_CLOSE:  # 如果执行结果为关闭
                    task_data.task_flag = TASK_DATA_CLOSE  # 设置标志位为关闭
                    self.close()  # 关闭
                if task_stage.next_stage is not None:  # 如果下阶段不为空
                    task_stage.to_next_stage(task_data)  # 调用下阶段
            if self.balancer is not None:  # 如果平衡器不为空
                suitable_interval = self.balancer.get_suitable_interval(process_interval, self)  # 设置适合的间隔
                if suitable_interval > 0:  # 如果间隔大于0
                    time.sleep(suitable_interval)  # 睡眠一段时间

    def start(self):
        p = Thread(target=self._run, args=())  # 调用线程与运行
        p.start()  # 开启
        self.result_worker = p  # 结果工作为p
        return p

    def put_task_data(self, task_data):  # 插入数据
        self.queue.put(task_data)  # 将工作数据插入到队列
        self._refresh_process_interval_scale()

    def _refresh_process_interval_scale(self):
        # 刷新过程间隔尺度
        self.process_interval_scale = max(self.queue.qsize(), 1)

    def product_task_data(self):  # 产生工作数据
        try:  # 尝试
            task_data = self.queue.get(block=True, timeout=1)  # 会阻止最多超时1秒
            self._refresh_process_interval_scale()
            return task_data  # 返回数据
        except Empty:
            return self.ignore_task_data  # 如果数据为空，返回没有数据

    def close(self):  # 关闭
        print(f'closing: {self}')  # 展示关闭的名称
        self.running = False  # 设置运行位为False

    def wait_for_end(self):
        self.result_worker.join()  # 加入线程
