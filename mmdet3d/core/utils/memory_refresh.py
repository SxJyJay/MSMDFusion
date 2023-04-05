from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class Refresh_Memory(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        pass

    def after_train_epoch(self, runner):
        # for MMDistributedDataParallel type of model
        runner.model.module.memory.refresh()

    def before_train_iter(self, runner):
        pass

    def after_train_iter(self, runner):
        pass