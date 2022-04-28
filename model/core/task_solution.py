from model.core.base_module import BaseModule, TaskStage, ModuleBalancer

class TaskSolution:
    # 任务的解决方案
    def __init__(self):
        self.modules = []  # 模型列表
        self.start_stage: TaskStage = TaskStage()  # 开始阶段
        self.current_stage: TaskStage = self.start_stage  # 当前阶段
        self.source_module = None  # 资源模型
        self.balancer = ModuleBalancer()  # 模块均衡器

    def set_source_module(self, source_module):
        # 设置资源模型
        source_module.balancer = self.balancer  # 平衡器
        self.source_module = source_module  # 资源模型
        source_module.task_stage = self.start_stage  # 开始阶段
        return self

    def set_next_module(self, next_module: BaseModule):
        # 设置下个模型
        next_module.balancer = self.balancer  # 平衡器
        next_stage = TaskStage()  # 下个阶段
        self.current_stage.next_module = lambda: next_module  # 当前阶段的下个模型(隐函数，只使用一次）
        self.current_stage.next_stage = next_stage  # 当前阶段的下个阶段
        self.current_stage = next_stage  # 当前阶段设为下个阶段
        self.modules.append(next_module)  # 添加模型
        return self

    def start(self):
        # 开始
        for module in self.modules:
            print(f'starting modules {module}')  # 打印开启的模型
            module.start()  # 遍历模型列表所有模型进行开启
        self.source_module.start()  # 资源模型开启

    def wait_for_end(self):
        # 等待结束
        self.source_module.wait_for_end()  # 等待结束
        for module in self.modules:
            module.wait_for_end()  # 遍历模型列表所有模型进行等待结束

    def close(self):
        # 关闭
        self.source_module.close()
        for module in self.modules:
            print(f'closing modules {module}')  # 打印关闭的模型
            module.close()  # 遍历模型列表所有模型进行关闭
